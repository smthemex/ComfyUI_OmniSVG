
from comfy_api.latest import ComfyExtension
from .OmniSVG_node import OmniSVG_Loader, OmniSVG_Sampler


class OmniSVG_Extension(ComfyExtension):
    async def get_node_list(self):
        return [OmniSVG_Loader, OmniSVG_Sampler]


async def comfy_entrypoint() -> OmniSVG_Extension:
    return OmniSVG_Extension()