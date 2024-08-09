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

import llm
import boto3


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

        response = bedrock.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(params)
        )

        response_body = json.loads(response.get("body").read())
        image_data = response_body.get("images")[0]
        image_bytes = base64.b64decode(image_data)
        image_filename = self.prompt_to_filename(prompt)

        with open(image_filename, "wb") as fp:
            fp.write(image_bytes)

        yield f"Result image written to: {image_filename}"
