"""VLM integration façade."""
class VLMService:
    @staticmethod
    def extract(model, api_key, prompt, image_paths, endpoint, timeout, schema):
        from homework_extractor import call_doubao
        return call_doubao(model, api_key, prompt, image_paths, endpoint, timeout, schema)

    @staticmethod
    def propose_structure(prompt, image_paths, endpoint, model, api_key="", timeout=180):
        """Return semantic structure only; caller must use OCR for geometry."""
        from .structure_vlm import PaddleOCRVLStructureClient, remove_model_geometry
        raw = PaddleOCRVLStructureClient(
            endpoint=endpoint, model=model, api_key=api_key, timeout=timeout,
        ).analyze(prompt, image_paths)
        return remove_model_geometry(raw)[0]
