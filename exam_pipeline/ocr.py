"""Lazy OCR façade."""
import copy
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence
from .contracts import OCRBlock


class OCRService:
    _cache = OrderedDict()
    _cache_lock = threading.Lock()
    _cache_limit = 512

    @classmethod
    def _key(cls, route, path, bbox, padding, engine, language):
        source = Path(path)
        try:
            stat = source.stat()
            fingerprint = (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
        except OSError:
            fingerprint = (str(source), None, None)
        box = tuple(round(float(value), 2) for value in (bbox or ()))
        return route, fingerprint, box, int(padding), str(engine), str(language)

    @classmethod
    def _cached(cls, key):
        with cls._cache_lock:
            value = cls._cache.get(key)
            if value is None:
                return None
            cls._cache.move_to_end(key)
            return copy.deepcopy(value)

    @classmethod
    def _store(cls, key, value):
        with cls._cache_lock:
            cls._cache[key] = copy.deepcopy(value)
            cls._cache.move_to_end(key)
            while len(cls._cache) > cls._cache_limit:
                cls._cache.popitem(last=False)

    @staticmethod
    def _metric(operation, started, path, cache_hit=False, status='SUCCESS', error_type=None):
        from .performance import get_active_collector
        collector = get_active_collector()
        if collector:
            collector.record(
                stage='ocr', provider='local_ocr', operation=operation,
                duration_seconds=time.monotonic() - started, paths=[Path(path)],
                cache_hit=cache_hit, status=status, error_type=error_type,
            )

    @staticmethod
    def recognize_page(path: Path, engine: str = "paddle",
                       language: str = "chi_sim+eng") -> List[OCRBlock]:
        started = time.monotonic()
        from homework_extractor import ocr_page
        try:
            result = ocr_page(Path(path), engine, language)
        except Exception as exc:
            OCRService._metric('ocr_page', started, path, status='FAILED',
                               error_type=type(exc).__name__)
            raise
        OCRService._metric('ocr_page', started, path)
        return result

    def recognize_crop(self, path: Path, bbox: Sequence[float], padding: int = 0,
                       engine: str = "paddle", language: str = "chi_sim+eng") -> List[OCRBlock]:
        import cv2
        import tempfile

        started = time.monotonic()
        key = self._key('crop', path, bbox, padding, engine, language)
        cached = self._cached(key)
        if cached is not None:
            self._metric('ocr_crop', started, path, cache_hit=True)
            return cached
        image = cv2.imread(str(path))
        if image is None or len(bbox) < 4:
            self._metric('ocr_crop', started, path)
            return []
        height, width = image.shape[:2]
        left, top, right, bottom = [int(round(value)) for value in bbox[:4]]
        left, top = max(0, left - padding), max(0, top - padding)
        right, bottom = min(width, right + padding), min(height, bottom + padding)
        if right <= left or bottom <= top:
            self._metric('ocr_crop', started, path)
            return []
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
                cv2.imwrite(handle.name, image[top:bottom, left:right])
                blocks = self.recognize_page(Path(handle.name), engine, language)
            result = [OCRBlock(block.text,
                               [block.bbox[0] + left, block.bbox[1] + top,
                                block.bbox[2] + left, block.bbox[3] + top],
                               block.confidence) for block in blocks]
        except Exception as exc:
            self._metric('ocr_crop', started, path, status='FAILED',
                         error_type=type(exc).__name__)
            raise
        self._store(key, result)
        self._metric('ocr_crop', started, path)
        return copy.deepcopy(result)

    def recognize_crop_high_resolution(self, path: Path, bbox: Sequence[float],
                                       padding: int = 8, scale: float = 2.5,
                                       engine: str = "paddle",
                                       language: str = "chi_sim+eng") -> List[OCRBlock]:
        """Re-detect a mixed prompt/answer crop and map boxes to page pixels."""
        import cv2
        import tempfile

        started = time.monotonic()
        scale = max(1.5, min(4.0, float(scale)))
        key = self._key('crop_highres:{:.2f}'.format(scale), path, bbox,
                        padding, engine, language)
        cached = self._cached(key)
        if cached is not None:
            self._metric('ocr_crop_highres', started, path, cache_hit=True)
            return cached
        image = cv2.imread(str(path))
        if image is None or len(bbox) < 4:
            self._metric('ocr_crop_highres', started, path)
            return []
        height, width = image.shape[:2]
        left, top, right, bottom = [int(round(value)) for value in bbox[:4]]
        left, top = max(0, left - padding), max(0, top - padding)
        right, bottom = min(width, right + padding), min(height, bottom + padding)
        if right <= left or bottom <= top:
            self._metric('ocr_crop_highres', started, path)
            return []
        crop = image[top:bottom, left:right]
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        try:
            with tempfile.NamedTemporaryFile(suffix='.png') as handle:
                cv2.imwrite(handle.name, enlarged)
                blocks = self.recognize_page(Path(handle.name), engine, language)
            result = []
            for block in blocks:
                mapped = [left + block.bbox[0] / scale,
                          top + block.bbox[1] / scale,
                          left + block.bbox[2] / scale,
                          top + block.bbox[3] / scale]
                mapped = [int(round(value)) for value in mapped]
                mapped[0], mapped[1] = max(0, mapped[0]), max(0, mapped[1])
                mapped[2], mapped[3] = min(width, mapped[2]), min(height, mapped[3])
                if mapped[0] < mapped[2] and mapped[1] < mapped[3]:
                    result.append(OCRBlock(block.text, mapped, block.confidence))
        except Exception as exc:
            self._metric('ocr_crop_highres', started, path, status='FAILED',
                         error_type=type(exc).__name__)
            raise
        self._store(key, result)
        self._metric('ocr_crop_highres', started, path)
        return copy.deepcopy(result)

    def recognize_line(self, path: Path, bbox, engine="paddle", language="chi_sim+eng"):
        """Recognition-only route for tiny glyphs rejected by text detection."""
        import cv2
        if engine not in {"paddle", "auto"}:
            return self.recognize_crop(path, bbox, padding=8, engine=engine, language=language)
        started = time.monotonic()
        key = self._key('line', path, bbox, 8, engine, language)
        cached = self._cached(key)
        if cached is not None:
            self._metric('ocr_line', started, path, cache_hit=True)
            return cached
        import homework_extractor as runtime
        image = cv2.imread(str(path))
        if image is None:
            self._metric('ocr_line', started, path)
            return []
        x1,y1,x2,y2 = [int(v) for v in bbox]
        crop = image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)]
        if not crop.size:
            self._metric('ocr_line', started, path)
            return []
        # Reuse the initialized recognition model; do not initialize a model
        # per slot. Unsupported runtimes retain the regular crop path.
        pipeline = getattr(runtime._PADDLE_OCR, "paddlex_pipeline", None)
        pipeline = getattr(pipeline, "_pipeline", pipeline)
        recognizer = getattr(pipeline, "text_rec_model", None)
        if recognizer is None:
            return self.recognize_crop(path,bbox,padding=8,engine=engine,language=language)
        crop = cv2.copyMakeBorder(crop,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))
        outputs = list(recognizer([crop]))
        result=[]
        for output in outputs:
            text = output.get("rec_text", "")
            score = output.get("rec_score")
            if text:
                result.append(OCRBlock(str(text),list(bbox),float(score) if score is not None else None))
        self._store(key, result)
        self._metric('ocr_line', started, path)
        return copy.deepcopy(result)
