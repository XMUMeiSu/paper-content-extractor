"""Teacher-to-student page registration."""
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


MIN_ECC_SCORE = 0.35
MIN_FEATURE_MATCHES = 10
MIN_FEATURE_INLIER_RATIO = 0.35
MIN_HOMOGRAPHY_INLIER_RATIO = 0.30


def _feature_register(reference, moving):
    """Return a moving->reference affine transform from local visual features."""
    import cv2
    import numpy as np

    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8)
    key_ref, desc_ref = detector.detectAndCompute(gray_ref, None)
    key_mov, desc_mov = detector.detectAndCompute(gray_mov, None)
    if desc_ref is None or desc_mov is None:
        return None, {"matches": 0, "inlier_ratio": 0.0, "reason": "ORB descriptors unavailable"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_mov, desc_ref, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2
            and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < MIN_FEATURE_MATCHES:
        return None, {"matches": len(good), "inlier_ratio": 0.0,
                      "reason": f"ORB matches below {MIN_FEATURE_MATCHES}"}
    source = np.float32([key_mov[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    target = np.float32([key_ref[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=3000, confidence=0.995, refineIters=20,
    )
    ratio = float(inliers.mean()) if inliers is not None and len(inliers) else 0.0
    meta = {"matches": len(good), "inliers": int(inliers.sum()) if inliers is not None else 0,
            "inlier_ratio": round(ratio, 5)}
    if matrix is not None and ratio >= MIN_FEATURE_INLIER_RATIO:
        meta["transform_type"] = "affine_partial"
        return matrix, meta

    # Phone photos commonly contain perspective distortion that a partial
    # affine transform cannot represent. Reuse the vetted feature matches for a
    # projective fallback instead of silently accepting a poor affine fit.
    homography, homography_inliers = cv2.findHomography(
        source.reshape(-1, 2), target.reshape(-1, 2),
        cv2.RANSAC, 4.0, maxIters=4000, confidence=0.995,
    )
    homography_ratio = float(homography_inliers.mean()) if (
        homography_inliers is not None and len(homography_inliers)
    ) else 0.0
    meta.update({
        "affine_inlier_ratio": round(ratio, 5),
        "homography_inliers": int(homography_inliers.sum())
        if homography_inliers is not None else 0,
        "homography_inlier_ratio": round(homography_ratio, 5),
    })
    if homography is None or homography_ratio < MIN_HOMOGRAPHY_INLIER_RATIO:
        meta["reason"] = (
            f"ORB affine/projective inlier ratios below "
            f"{MIN_FEATURE_INLIER_RATIO:.2f}/{MIN_HOMOGRAPHY_INLIER_RATIO:.2f}"
        )
        return None, meta
    meta["inliers"] = meta["homography_inliers"]
    meta["inlier_ratio"] = meta["homography_inlier_ratio"]
    meta["transform_type"] = "homography"
    return homography, meta


def register_page(reference_path: Path, moving_path: Path, output_path: Path) -> Tuple[Path, Dict[str, Any]]:
    import cv2
    import numpy as np

    reference = cv2.imread(str(reference_path))
    moving = cv2.imread(str(moving_path))
    if reference is None or moving is None:
        return Path(moving_path), {"status": "FAILED", "reason": "image unreadable"}
    height, width = reference.shape[:2]
    resized = cv2.resize(moving, (width, height))
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    status, score, method, reason = "REGISTERED", 0.0, "ecc_affine", ""
    try:
        score, warp = cv2.findTransformECC(gray_ref, gray_mov, warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5))
        if float(score) < MIN_ECC_SCORE:
            registered = resized
            status = "FAILED_LOW_CONFIDENCE"
            reason = f"ECC score {float(score):.5f} below {MIN_ECC_SCORE:.2f}"
        else:
            registered = cv2.warpAffine(
                resized, warp, (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
    except cv2.error as exc:
        registered = resized
        status = "FAILED"
        reason = str(exc).splitlines()[0]
    feature_meta = {}
    if status != "REGISTERED":
        feature_warp, feature_meta = _feature_register(reference, resized)
        if feature_warp is not None:
            if feature_warp.shape == (3, 3):
                registered = cv2.warpPerspective(
                    resized, feature_warp, (width, height), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                feature_method = "orb_homography"
            else:
                registered = cv2.warpAffine(
                    resized, feature_warp, (width, height), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                feature_method = "orb_ransac"
            warp = feature_warp
            score = feature_meta["inlier_ratio"]
            status, method, reason = "REGISTERED", feature_method, ""
        else:
            method = "unregistered"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), registered, [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta = {"status": status, "score": round(float(score), 5), "matrix": warp.tolist(),
            "method": method, "reference": str(reference_path),
            "source": str(moving_path), "output": str(output_path)}
    if feature_meta:
        meta["feature_audit"] = feature_meta
    if status != "REGISTERED":
        meta["reason"] = reason
    return output_path, meta


def register_pages(reference_paths: Sequence[Path], moving_paths: Sequence[Path],
                   output_dir: Path, document_id: str) -> Tuple[List[Path], List[Dict[str, Any]]]:
    outputs, metadata = [], []
    for index, moving in enumerate(moving_paths, 1):
        if index <= len(reference_paths):
            target = Path(output_dir) / document_id / f"page_{index:02d}.jpg"
            output, meta = register_page(reference_paths[index - 1], moving, target)
        else:
            output, meta = Path(moving), {"status": "FAILED", "reason": "missing teacher page"}
        outputs.append(output)
        metadata.append(meta)
    return outputs, metadata
