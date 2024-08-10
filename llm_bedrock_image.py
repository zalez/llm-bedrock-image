"""
llm-bedrock-image

A plugin for Simon Willison’s llm CLI utility: https://llm.datasette.io/
This plugin adds support for image generation models on Amazon Bedrock.

See the llm Plugins documentation on how to add this plugin to your installation of llm:
https://llm.datasette.io/en/stable/plugins/index.html

See the Amazon Bedrock documentation for more information on models and parameters.
https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html
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


# Constants

MAX_SEED = 2147483646
MAX_NUMBER_OF_IMAGES = 5


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
    can_stream = True  # Yield updates while saving images if streaming.

    class Options(llm.Options):
        bedrock_image_open: Optional[Union[str, bool]] = Field(
            description='Whether to automatically open images after generating (true, false, 0, 1).',
            default=True
        )
        bedrock_image_negative_prompt: Optional[str] = Field(
            description='Additional prompt with things to avoid generating.',
            default=None
        )
        bedrock_image_seed: Optional[int] = Field(
            description='Use to control and reproduce results. Determines the initial noise setting. Use the same ' +
                        'seed and the same settings as a previous run to allow inference to create a similar image.',
            default=None
        )
        bedrock_image_number_of_images: Optional[int] = Field(
            description='Number of images to generate.',
            default=None
        )

        @field_validator('bedrock_image_open')
        def validate_bedrock_image_open(cls, value):
            if value is None:
                return None
            if str(value).lower() not in ['true', 'false', '0', '1']:
                raise ValueError('bedrock_image_open must be one of true, false, 0, 1.')
            return str(value).lower() in ['true', '1']
        @field_validator('bedrock_image_negative_prompt')
        def validate_bedrock_image_negative_prompt(cls, value):
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError('bedrock_image_negative_prompt must be a string.')
            return value
        @field_validator('bedrock_image_seed')
        def validate_bedrock_image_seed(cls, value):
            if value is None:
                return None
            if not isinstance(value, int):
                raise ValueError('bedrock_image_seed must be an integer.')
            if value < 0 or value > MAX_SEED:
                raise ValueError(f'bedrock_image_seed must be between 0 and {MAX_SEED}.')
            return value
        @field_validator('bedrock_image_number_of_images')
        def validate_bedrock_image_number_of_images(cls, value):
            if value is None:
                return None
            if not isinstance(value, int):
                raise ValueError('bedrock_image_number_of_images must be an integer.')
            if value < 1 or value > MAX_NUMBER_OF_IMAGES:
                raise ValueError(f'bedrock_image_number_of_images must be between 1 and {MAX_NUMBER_OF_IMAGES}.')
            return value

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

        params= {
            'taskType': 'TEXT_IMAGE',
            'textToImageParams': {
                'text': prompt.prompt
            },
            'imageGenerationConfig': {}
        }
        if prompt.options.bedrock_image_negative_prompt:
            params["textToImageParams"]["negativeText"] = prompt.options.bedrock_image_negative_prompt
        if prompt.options.bedrock_image_seed:
            params['imageGenerationConfig']['seed'] = prompt.options.bedrock_image_seed
        if prompt.options.bedrock_image_number_of_images:
            params['imageGenerationConfig']['numberOfImages'] = prompt.options.bedrock_image_number_of_images

        return params

    @staticmethod
    def prompt_to_filename(prompt):
        """
        Generate a file name out of the given prompt object by replacing any non-alphanumeric
        character including spaces with underscore, then eliminating multiple underscores.
        :param prompt: A llm prompt object.
        :return: A file name string.
        """
        base = ""
        ext = ".png"
        for c in prompt.prompt:
            if c.isalnum():
                base += c
            else:
                base += "_"

        while "__" in base:
            base = base.replace("__", "_")

        base = base.strip('_')

        # Avoid overwriting existing files with the same name.
        i = 1
        path = base + ext
        while os.path.exists(path):
            path = base + f"_{i}" + ext
            i += 1

        return path

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
        :param conversation: A llm conversation object with the conversation history (if any) (unused).
        :return: None. Use yield to return results.
        """
        params = self.prompt_to_titan_params(prompt)
        bedrock = boto3.client("bedrock-runtime")

        if stream:
            yield "Generating image...\n"

        bedrock_response = bedrock.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(params)
        )

        response_body = json.loads(bedrock_response.get("body").read())

        # Process each image and remove them from the body, so we can log the rest without the
        # bulky image data.

        file_names = []
        for i, image_data in enumerate(response_body.get("images")):
            image_bytes = base64.b64decode(image_data)
            image_filename = self.prompt_to_filename(prompt)
            with open(image_filename, "wb") as fp:
                fp.write(image_bytes)

            if prompt.options.bedrock_image_open:
                self.open_file(image_filename)

            response_body['images'][i] = f'<image data {i + 1}>'

            file_names.append(image_filename)
            if stream:
                yield f"Image written to: {image_filename}\n"

        # Log the rest of the response body in case there's more useful information in it.
        response.response_json = json.dumps(response_body)
        if not stream:
            if len(file_names) == 1:
                yield 'Image written to: ' + file_names[0] + '\n'
            else:
                yield 'Images written to:\n' + '\n    '.join(file_names) + '\n'
