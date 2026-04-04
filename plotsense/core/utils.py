import base64
import getpass
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Union, cast


def prompt_for_api_key(
    service_name: str, service_link: str, interactive: bool = True,
    skip_if_missing: bool = False
) -> Optional[str]:
    """Prompt user for API key.

    Args:
        service_name: Name of the service/provider
        service_link: Link to get the API key
        interactive: Whether to prompt (vs raise immediately)
        skip_if_missing: Whether pressing Enter without a value is allowed

    Returns:
        API key string, or None if user skips and skip_if_missing=True

    Raises:
        ValueError if key not provided and either not interactive or skip_if_missing=False
    """
    if not interactive:
        if skip_if_missing:
            return None
        raise ValueError(
            f"{service_name.upper()} API key is required. "
            f"Set it in the environment or pass it as an argument. "
            f"You can get it at {service_link}"
        )

    try:
        print(f"⚙️  {service_name.upper()} API key not found.")
        print(f"🔗  Get it at {service_link}")
        if skip_if_missing:
            prompt_text = f"Enter {service_name.upper()} API key (or press Enter to skip): "
        else:
            prompt_text = f"Enter {service_name.upper()} API key: "
        # Use getpass to hide input (avoid key appearing in logs/terminal)
        key = getpass.getpass(prompt_text).strip()

        if not key and skip_if_missing:
            return None
        if not key:
            raise ValueError(f"{service_name.upper()} API key is required.")
        return key
    except (EOFError, OSError):
        if skip_if_missing:
            return None
        raise ValueError(f"{service_name.upper()} API key is required (get it at {service_link})")


def save_plot_to_image(
    plot_object: Union[Figure, Axes],
    output_path: str = "temp_plot.jpg"
) -> str:
    """Save a matplotlib Figure or Axes object to a JPEG image file."""
    if isinstance(plot_object, Axes):
        fig = plot_object.figure
    else:
        fig = plot_object
    cast(Figure, fig).savefig(
        output_path, format='jpeg', dpi=100, bbox_inches='tight'
    )
    return output_path


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
