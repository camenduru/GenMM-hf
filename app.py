import json
import time

import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from dataset.tracks_motion import TracksMotion
from GPS import GPS
import gradio as gr


def _synthesis(synthesis_setting, motion_data):
    model = GPS(
        init_mode=f"random_synthesis/{synthesis_setting['frames']}",
        noise_sigma=synthesis_setting["noise_sigma"],
        coarse_ratio=0.2,
        pyr_factor=synthesis_setting["pyr_factor"],
        num_stages_limit=-1,
        silent=True,
        device="cpu",
    )

    synthesized_motion = model.run(
        motion_data,
        mode="match_and_blend",
        ext={
            "criteria": {
                "type": "PatchCoherentLoss",
                "patch_size": synthesis_setting["patch_size"],
                "stride": synthesis_setting["stride"]
                if "stride" in synthesis_setting.keys()
                else 1,
                "loop": synthesis_setting["loop"],
                "coherent_alpha": synthesis_setting["alpha"]
                if synthesis_setting["completeness"]
                else None,
            },
            "optimizer": "match_and_blend",
            "num_itrs": synthesis_setting["num_steps"],
        },
    )

    return synthesized_motion


def synthesis(data):
    data = json.loads(data)
    # create track object
    data["setting"]["coarse_ratio"] = -1
    motion_data = TracksMotion(data["tracks"], scale=data["scale"])
    start = time.time()
    synthesized_motion = _synthesis(data["setting"], [motion_data])
    end = time.time()
    data["time"] = end - start
    data["tracks"] = motion_data.parse(synthesized_motion)

    return data


intro = """
<h1 style="text-align: center;">
   Example-based Motion Synthesis via Generative Motion Matching
</h1>
<h3 style="text-align: center; margin-bottom: 7px;">
    <a href="http://weiyuli.xyz/GenMM" target="_blank">Project Page</a> | <a href="https://huggingface.co/papers/2306.00378" target="_blank">Paper</a> | <a href="https://github.com/wyysf-98/GenMM" target="_blank">Code</a> 
</h3>
"""

with gr.Blocks() as demo:
    gr.HTML(intro)
    gr.HTML(
        """<iframe src="/GenMM_demo/" width="100%" height="700px" style="border:none;">"""
    )
    json_in = gr.JSON(visible=False)
    json_out = gr.JSON(visible=False)
    btn = gr.Button("Synthesize", visible=False)
    btn.click(synthesis, inputs=[json_in], outputs=[json_out], api_name="predict")


app = FastAPI()

static_dir = Path("./GenMM_demo")
app.mount("/GenMM_demo", StaticFiles(directory=static_dir, html=True), name="static")
app = gr.mount_gradio_app(app, demo, path="/")


# serve the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
