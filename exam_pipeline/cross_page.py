"""Compatibility façade for cross-page stitching implemented by the CLI module."""


def analyze_page_exit(page, sections):
    from homework_extractor import analyze_page_exit as implementation
    return implementation(page, sections)


def should_stitch_page(exit_context, sections):
    from homework_extractor import should_stitch_page as implementation
    return implementation(exit_context, sections)


def stitch_page_sections(package, sections, exit_context):
    from homework_extractor import stitch_page_sections as implementation
    return implementation(package, sections, exit_context)


__all__ = ["analyze_page_exit", "should_stitch_page", "stitch_page_sections"]
