"""Pipeline orchestration façade."""
class ExamPipeline:
    @staticmethod
    def run(*args, **kwargs):
        from homework_extractor import process
        return process(*args, **kwargs)
