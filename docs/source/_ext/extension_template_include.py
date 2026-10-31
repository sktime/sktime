"""Sphinx extension to include sktime extension template content in LLM prompts."""

from pathlib import Path

from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)


class ExtensionTemplateIncludeDirective(SphinxDirective):
    """Directive to include content from extension template files."""

    required_arguments = 1  # Template file name
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        "lines": unchanged,
        "start-after": unchanged,
        "end-before": unchanged,
    }

    def run(self):
        """Run the directive to include extension template content."""
        env = self.state.document.settings.env

        # Get the template file name
        template_name = self.arguments[0].strip()

        # More robust way to find the extension templates directory
        # First try relative to source directory
        template_path = (
            Path(env.srcdir) / ".." / ".." / "extension_templates" / template_name
        )

        # If that doesn't work, try to find project root by looking for pyproject.toml
        if not template_path.exists():
            # Go up the directory tree from the source dir until we find pyproject.toml
            src_path = Path(env.srcdir)
            project_root = src_path.parent.parent.absolute()
            template_path = project_root / "extension_templates" / template_name

        if not template_path.exists():
            # Try using the current working directory as fallback
            fallback_path = Path("extension_templates") / template_name
            if fallback_path.exists():
                template_path = fallback_path
            else:
                logger.warning(f"Extension template file not found: {template_path}")
                error = self.state_machine.reporter.error(
                    f"Extension template file not found: {template_path}",
                    nodes.literal_block(self.block_text, self.block_text),
                    line=self.lineno,
                )
                return [error]

        # Read the template file
        try:
            content = template_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(
                f"Error reading extension template file {template_path}: {e}"
            )
            error = self.state_machine.reporter.error(
                f"Error reading extension template file: {template_path}",
                nodes.literal_block(self.block_text, self.block_text),
                line=self.lineno,
            )
            return [error]

        # Apply line filtering if requested
        if "lines" in self.options:
            line_spec = self.options["lines"]
            try:
                line_parts = line_spec.strip().split("-")
                if len(line_parts) != 2:
                    raise ValueError(
                        f"Invalid line range format: {line_spec}. Expected: start-end"
                    )
                start_line, end_line = map(int, line_parts)
                lines = content.split("\n")
                # Ensure valid range
                start_line = max(1, start_line)
                end_line = min(len(lines), end_line)
                content = "\n".join(lines[start_line - 1 : end_line])
            except ValueError as e:
                logger.warning(f"Error parsing lines option: {e}")
                error = self.state_machine.reporter.error(
                    f"Error parsing lines option '{line_spec}': {str(e)}",
                    nodes.literal_block(self.block_text, self.block_text),
                    line=self.lineno,
                )
                return [error]
        elif "start-after" in self.options or "end-before" in self.options:
            lines = content.split("\n")
            start_after = self.options.get("start-after", "")
            end_before = self.options.get("end-before", "")

            start_idx = 0
            if start_after:
                for i, line in enumerate(lines):
                    if start_after in line:
                        start_idx = i + 1
                        break

            end_idx = len(lines)
            if end_before:
                for i, line in enumerate(lines[start_idx:], start_idx):
                    if end_before in line:
                        end_idx = i
                        break

            content = "\n".join(lines[start_idx:end_idx])

        # Create a literal block node with the content
        literal_node = nodes.literal_block(content, content)
        literal_node["language"] = "python"

        return [literal_node]


def setup(app):
    """Set up the extension."""
    app.add_directive("extension-template-include", ExtensionTemplateIncludeDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
