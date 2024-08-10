"""
llm-bedrock-image

A plugin for Simon Willison’s llm CLI utility: https://llm.datasette.io/
This plugin adds support for image generation models on Amazon Bedrock.

See the llm Plugins documentation on how to add this plugin to your installation of llm:
https://llm.datasette.io/en/stable/plugins/index.html
"""


# Imports

import json
import base64
import subprocess
import os
import platform
from typing import Optional, Union

import llm
import boto3
from pydantic import field_validator, Field

# Functions

@llm.hookimpl
def register_models(register):
    """
    Register the models we support with llm.
    :param register: llm’s register function for registering plugins.
    :return: None
    """
    register(
        BedrockImage("amazon.titan-image-generator-v2:0"),
        aliases=("bedrock-image-titan-v2", "bedrock-image-titan", "bit"),
    )
    # TODO: Add Titan v1 as well.
    # TODO: Add Stable diffusion.


# Classes

class BedrockImage(llm.Model):
    can_stream = False  # We don't support streaming.

    class Options(llm.Options):
        bedrock_image_open: Optional[Union[str, bool]] = Field(
            description="Whether to automatically open images after generating (true, false, 0, 1).",
            default="true"
        )
        @field_validator("bedrock_image_open")
        def validate_bedrock_image_open(cls, value):
            if value is None:
                return None
            if str(value).lower() not in ['true', 'false', '0', '1']:
                raise ValueError("bedrock_image_open must be one of true, false, 0, 1.")
            return str(value).lower() in ['true', '1']

    def __init__(self, model_id):
        self.model_id = model_id

    @staticmethod
    def prompt_to_titan_params(prompt):
        """
        Generate a Bedrock Titan image parameters dict based on the given prompt.
        :param prompt: A llm prompt object.
        :return: A dict of Bedrock Titan image parameters.
        """
        if len(prompt.prompt) > 512:
            raise ValueError(
                "Prompt is too long. Maximum prompt length for Amazon Titan Image " +
                "Generator G1 models is 512 characters."
            )

        return {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt.prompt
            }
        }

    @staticmethod
    def prompt_to_filename(prompt):
        """
        Generate a file name out of the given prompt object by replacing any non-alphanumeric
        character including spaces with underscore, then eliminating multiple underscores.
        :param prompt: A llm prompt object.
        :return: A file name string.
        """
        result = ""
        for c in prompt.prompt:
            if c.isalnum():
                result += c
            else:
                result += "_"

        while "__" in result:
            result = result.replace("__", "_")

        result = result.strip('_')

        result += ".png"

        return result

    @staticmethod
    def open_file(path):
        """
        Open the given file path on the user’s OS with its default application.
        See: https://stackoverflow.com/questions/434597/open-document-with-default-os-application-in-python-both-in-windows-and-mac-os

        :param path: The path to a file to be opened.
        :return: None
        """
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', path))
        elif platform.system() == 'Windows':  # Windows
            # os.startfile(path) # Replaced by Amazon Q Developer with the below line to improve security.
            subprocess.run([filepath], shell=False, check=True, capture_output=True, text=True)
        else:  # linux variants
            subprocess.call(('xdg-open', path))

    def execute(self, prompt, stream, response, conversation):
        """
        Take the prompt and run it through our model. Return a response in streaming
        mode (if needed).

        We ignore the conversation, since text to image use cases don't have a concept of a
        conversation.

        :param prompt: The prompt object we’re given from llm.
        :param stream: Whether we're streaming output or not.
        :param response: A llm response object for storing response details.
        :param conversation: A llm conversation object with the conversation history (if any).
        :return: None. Use yield to return results.
        """

        params = self.prompt_to_titan_params(prompt)
        bedrock = boto3.client("bedrock-runtime")

        bedrock_response = bedrock.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(params)
        )

        response_body = json.loads(bedrock_response.get("body").read())
        image_data = response_body.get("images")[0]
        image_bytes = base64.b64decode(image_data)
        image_filename = self.prompt_to_filename(prompt)

        # Remove image from response body and log the rest.
        for i in range(len(response_body['images'])):
            response_body['images'][i] = f'<image data {i + 1}>'
        response.response_json = json.dumps(response_body)

        with open(image_filename, "wb") as fp:
            fp.write(image_bytes)

        if prompt.options.bedrock_image_open:
            self.open_file(image_filename)

        yield f"Result image written to: {image_filename}"
