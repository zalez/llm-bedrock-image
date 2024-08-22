"""
llm-bedrock-titan-image

A plugin for Simon Willison’s LLM CLI utility: https://llm.datasette.io/
This plugin adds support for Amazon Titan Image Generator G1 image generation models on Amazon Bedrock.

See the llm Plugins documentation on how to add this plugin to your installation of llm:
https://llm.datasette.io/en/stable/plugins/index.html

See the Amazon Bedrock documentation for more information on models and parameters.
https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html
"""


# Imports

from functools import cached_property
import json
import base64
import subprocess
import os
import platform
from typing import Optional, Union
import concurrent.futures
import re
from io import BytesIO

import requests
import llm
import boto3
from pydantic import field_validator, Field
import webcolors
from PIL import Image


# Constants

DEBUG = os.environ.get('DEBUG', 'false').lower() in ['true', '1']

# Model options and defaults

MAX_PROMPT_LENGTH = 512

MIN_NUMBER_OF_IMAGES = 1
MAX_NUMBER_OF_IMAGES = 5
DEFAULT_NUMBER_OF_IMAGES = 1

MIN_CFG_SCALE = 1.1
MAX_CFG_SCALE = 10.0
DEFAULT_CFG_SCALE = 8.0

# (width, height),
# see: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
IMAGE_SIZES = [
    (1024, 1024),
    (768, 768),
    (512, 512),
    (768, 1152),
    (384, 576),
    (1152, 768),
    (576, 384),
    (768, 1280),
    (384, 640),
    (1280, 768),
    (640, 384),
    (896, 1152),
    (448, 576),
    (1152, 896),
    (576, 448),
    (768, 1408),
    (384, 704),
    (1408, 768),
    (704, 384),
    (640, 1408),
    (320, 704),
    (1408, 640),
    (704, 320),
    (1152, 640),
    (1173, 640)
]
IMAGE_HEIGHTS = list(set([i[1] for i in IMAGE_SIZES]))
IMAGE_WIDTHS = list(set([i[0] for i in IMAGE_SIZES]))
DEFAULT_IMAGE_SIZE = (1024, 1024)

MIN_SEED = 0
MAX_SEED = 2147483646
DEFAULT_SEED = 0

MIN_CONTROL_STRENGTH = 0.0
MAX_CONTROL_STRENGTH = 1.0
DEFAULT_CONTROL_STRENGTH = 0.7

MIN_SIMILARITY_STRENGTH = 0.2
MAX_SIMILARITY_STRENGTH = 1.0
DEFAULT_SIMILARITY_STRENGTH = 0.7

MIN_COLORS = 1
MAX_COLORS = 10

MIN_INPUT_IMAGE_LENGTH = 256
MAX_INPUT_IMAGE_LENGTH = 1408
SUPPORTED_INPUT_IMAGE_FORMATS = ['png', 'jpg', 'jpeg']

MAX_INPUT_IMAGES_FOR_VARIATION = 5

# Pricing information

# Taken from looking at: https://aws.amazon.com/bedrock/pricing/
# There are no guarantees that this will work in the future.
AWS_BEDROCK_PRICING_URL = 'https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/bedrock/USD/current/bedrock.json'
AWS_LOCATIONS_URL = 'https://b0.p.awsstatic.com/locations/1.0/aws/current/locations.json'

# Map model IDs to pricing table model names.
MODEL_ID_TO_MODEL_NAME = {
    'amazon.titan-image-generator-v1': 'Titan Image Generator G1',
    'amazon.titan-image-generator-v2:0': 'Titan Image Generator V2',  # G1 V2 is correct, but the price list needs this.
}

# Map model names to supported additional options.
MODEL_ID_TO_EXTRA_OPTIONS = {
    'amazon.titan-image-generator-v1': [],
    'amazon.titan-image-generator-v2:0': [
        'image', 'control_mode', 'control_strength'
    ]
}

# Map model names to supported task types.
MODEL_ID_TO_TASK_TYPES = {
    'amazon.titan-image-generator-v1': [
        'TEXT_IMAGE',
        'INPAINTING',
        'OUTPAINTING',
        'IMAGE_VARIATION',
        # 'DETECT_GENERATED_CONTENT'
    ],
    'amazon.titan-image-generator-v2:0': [
        'TEXT_IMAGE',
        'INPAINTING',
        'OUTPAINTING',
        'IMAGE_VARIATION',
        'COLOR_GUIDED_GENERATION',
        'BACKGROUND_REMOVAL',
    ]
}

TASKS = list({task for tasks in MODEL_ID_TO_TASK_TYPES.values() for task in tasks})
TASKS_WITH_NO_PROMPT_NEEDED = [
    'BACKGROUND_REMOVAL',
    'INPAINTING'
    # 'DETECT_GENERATED_CONTENT'
]
DEFAULT_TASK = 'TEXT_IMAGE'


# Functions

def print_debug(name, value):
    """
    Print a JSON dump of the given value under the given name, only if DEBUG is set.
    """
    if DEBUG:
        print(f'{name}:')
        print(json.dumps(value, sort_keys=True, indent=2))


@llm.hookimpl
def register_models(register):
    """
    Register the models we support with llm.
    :param register: llm’s register function for registering plugins.
    :return: None
    """
    register(
        BedrockTitanImage('amazon.titan-image-generator-v1'),
        aliases=('amazon-titan-image-v1', 'ati1'),
    )
    register(
        BedrockTitanImage('amazon.titan-image-generator-v2:0'),
        aliases=('amazon-titan-image-v2', 'amazon-titan-image', 'ati2', 'ati'),
    )


# Classes

class BedrockTitanImage(llm.Model):
    """
    This class handles Amazon Titan Image Generator models on Amazon Bedrock.
    """
    can_stream = True  # Yield updates while saving images if streaming.

    # Model and other options we support.
    # We assume that these are scoped to our class only, so we can use simple names.
    class Options(llm.Options):
        region: Optional[Union[str, bool]] = Field(
            description='The AWS region to use for this model. Overrides the AWS_DEFAULT_REGION environment, ' +
                        'if set. If none is configured and auto_region is true, this plugin will automatically ' +
                        'choose a supported and enabled region. If this is set, the auto_region option is ignored.',
            default=None
        )
        task: str = Field(
            description='The task to perform. By default, it is TEXT_IMAGE for generating an image from text.',
            default='TEXT_IMAGE'
        )
        auto_region: Optional[Union[str, bool]] = Field(
            description='Choose a supported AWS Region automatically if none is configured or the model isn’t ' +
                        'available/accessible there (true, false, on, off, 0, 1).',
            default=True
        )
        auto_open: Optional[Union[str, bool]] = Field(
            description='Open images after automatically after generating (true, false, on, off, 0, 1).',
            default=True
        )
        negative_prompt: Optional[str] = Field(
            description='Additional prompt with things to avoid generating.',
            default=None
        )
        number_of_images: Optional[int] = Field(
            description='The number of images to generate.',
            default=DEFAULT_NUMBER_OF_IMAGES
        )
        cfg_scale: Optional[Union[int, float]] = Field(
            description='Specifies how strongly the generated image should adhere to the prompt. Use a lower value ' +
                        'to introduce more randomness in the generation',
            default=DEFAULT_CFG_SCALE
        )
        height: Optional[int] = Field(
            description='The height of the image in pixels.',
            default=DEFAULT_IMAGE_SIZE[1]
        )
        width: Optional[int] = Field(
            description='The width of the image in pixels.',
            default=DEFAULT_IMAGE_SIZE[0]
        )
        seed: Optional[int] = Field(
            description='Use to control and reproduce results. Determines the initial noise setting. Use the same ' +
                        'seed and the same settings as a previous run to allow inference to create a similar image.',
            default=DEFAULT_SEED
        )
        quality: Optional[str] = Field(
            description='The quality setting to use. Can be ’standard’ or ’premium’.',
            default='standard'
        )
        image: Optional[str] = Field(
            description='File system path to an input image used for certain task types.',
            default=None
        )
        images: Optional[str] = Field(
            description='1 - 5 comma-separated file system paths to input image used for certain task types.',
            default=None
        )
        resize_image: Optional[Union[str, bool]] = Field(
            description="Automatically resize the image if it doesn't conform to supported size constraints.",
            default=True
        )
        mask_prompt: Optional[str] = Field(
            description='A text prompt that defines the mask to use for INPAINTING and OUTPAINTING tasks.',
            default=None
        )
        mask_image: Optional[str] = Field(
            description='A mask image that defines which pixels to modify (RGB: 0, 0, 0) or not (RGB: 255, 255, 255).' +
                        'To be used win INPAINTING or OUTPAINTING tasks as an alternative to mask prompts.',
            default=None
        )
        outpainting_mode: Optional[str] = Field(
            description='Specifies whether to allow modification of the pixels inside the mask or not. Can be ' +
                        'DEFAULT or PRECISE.',
            default='DEFAULT'
        )
        similarity_strength: Optional[Union[int, float]] = Field(
            description='Specifies how similar the generated image should be to the input image(s) for ' +
                        'IMAGE_VARIATION tasks. Use a lower value to introduce more randomness in the generation.',
            default=DEFAULT_SIMILARITY_STRENGTH
        )
        control_mode: Optional[str] = Field(
            description='What type of image conditioning to use. Can be CANNY_EDGE or or SEGMENTATION.',
            default='CANNY_EDGE'
        )
        control_strength: Optional[Union[int, float]] = Field(
            description='How similar the layout and composition of the generated image should be to the condition' +
                        'image.',
            default=DEFAULT_CONTROL_STRENGTH
        )
        colors: Optional[str] = Field(
            description='A list of hex color codes for conditioning the colors used when generating the image.',
            default=None
        )

        @field_validator('region')
        def validate_region(cls, value):
            if value is None:
                return None
            bedrock_regions = boto3.Session().get_available_regions('bedrock')
            if str(value).lower() not in bedrock_regions:
                raise ValueError(f'region must be one of the supported Amazon Bedrock regions: {bedrock_regions}.')
            return str(value).lower()

        @field_validator('task')
        def validate_task(cls, value):
            candidate_tasks = [t for t in TASKS if value.upper() in t]
            if not candidate_tasks:
                raise ValueError(f'task needs to be one of, or an unambiguous substring of: {TASKS}')
            if len(candidate_tasks) > 1:
                raise ValueError(f'task is ambiguous, it matches more than one task: {candidate_tasks}')
            return candidate_tasks[0]

        @field_validator('auto_region')
        def validate_auto_region(cls, value):
            if value is None:
                return True  # Default
            if str(value).lower() not in ['true', 'false', 'on', 'off', '0', '1']:
                raise ValueError('auto_region must be one of true, false, on, off, 0, 1.')
            return str(value).lower() in ['true', 'on', '1']

        @field_validator('auto_open')
        def validate_auto_open(cls, value):
            if value is None:
                return True  # Default
            if str(value).lower() not in ['true', 'false', 'on', 'off', '0', '1']:
                raise ValueError('auto_open must be one of true, false, on, off, 0, 1.')
            return str(value).lower() in ['true', 'on', '1']

        @field_validator('negative_prompt')
        def validate_negative_prompt(cls, value):
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError('negative_prompt must be a string.')
            return value

        @field_validator('number_of_images')
        def validate_number_of_images(cls, value):
            if value is None:
                return None
            if not isinstance(value, int):
                raise ValueError('number_of_images must be an integer.')
            if value < MIN_NUMBER_OF_IMAGES or value > MAX_NUMBER_OF_IMAGES:
                raise ValueError(
                    f'number_of_images must be between {MIN_NUMBER_OF_IMAGES} and {MAX_NUMBER_OF_IMAGES}.'
                )
            return value

        @field_validator('cfg_scale')
        def validate_cfg_scale(cls, value):
            if value is None:
                return None
            if not isinstance(value, (int, float)):
                raise ValueError('cfg_scale must be an integer or float.')
            if value < MIN_CFG_SCALE or value > MAX_CFG_SCALE:
                raise ValueError(f'cfg_scale must be between {MIN_CFG_SCALE} and {MAX_CFG_SCALE}.')
            return value

        @field_validator('height')
        def validate_height(cls, value):
            if value is None:
                return None
            if not isinstance(value, int):
                raise ValueError('height must be an integer.')
            if value not in IMAGE_HEIGHTS:
                raise ValueError(f'height must be one of: {', '.join(IMAGE_HEIGHTS)}.')
            return value

        @field_validator('width')
        def validate_width(cls, value):
            if value is None:
                return None
            if not isinstance(value, int):
                raise ValueError('width must be an integer.')
            if value not in IMAGE_WIDTHS:
                raise ValueError(f'width must be one of: {', '.join(IMAGE_WIDTHS)}.')
            return value

        @field_validator('seed')
        def validate_seed(cls, value):
            if value is None:
                return None
            if not isinstance(value, int):
                raise ValueError('seed must be an integer.')
            if value < MIN_SEED or value > MAX_SEED:
                raise ValueError(f'seed must be between {MIN_SEED} and {MAX_SEED}.')
            return value

        @field_validator('quality')
        def validate_quality(cls, value):
            if value is None:
                return None
            if str(value).lower() not in ['standard', 'premium']:
                raise ValueError('quality must be one of [standard|premium].')
            return str(value).lower()

        @field_validator('image')
        def validate_image(cls, value):
            if value is None:
                return None
            path = os.path.expanduser(value)
            if not os.path.exists(path):
                raise ValueError('image must be a valid file path.')
            return path

        @field_validator('images')
        def validate_images(cls, value):
            if value is None:
                return None
            for i in value.split(','):
                path = os.path.expanduser(i.strip())
                if not os.path.exists(path):
                    raise ValueError(f'Image: {path} is not a valid file path.')
            return value

        @field_validator('resize_image')
        def validate_resize_image(cls, value):
            if value is None:
                return True  # Default
            if str(value).lower() not in ['true', 'false', 'on', 'off', '0', '1']:
                raise ValueError('resize_image must be one of true, false, on, off, 0, 1.')
            return str(value).lower() in ['true', 'on', '1']

        @field_validator('mask_prompt')
        def validate_mask_prompt(cls, value):
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError('mask_prompt must be a string.')
            return value

        @field_validator('mask_image')
        def validate_mask_image(cls, value):
            if value is None:
                return None
            path = os.path.expanduser(value)
            if not os.path.exists(path):
                raise ValueError('image must be a valid file path.')
            return path

        @field_validator('outpainting_mode')
        def validate_outpainting_mode(cls, value):
            if value is None:
                return None
            if str(value).upper() not in ['DEFAULT', 'PRECISE']:
                raise ValueError('outpainting_mode must be one of [DEFAULT|PRECISE].')
            return str(value).upper()

        @field_validator('control_mode')
        def validate_control_mode(cls, value):
            if value is None:
                return None
            if str(value).upper() not in ['CANNY_EDGE', 'SEGMENTATION']:
                raise ValueError('control_mode must be one of [CANNY_EDGE|SEGMENTATION].')
            return str(value).upper()

        @field_validator('control_strength')
        def validate_control_strength(cls, value):
            if value is None:
                return None
            if not isinstance(value, (int, float)):
                raise ValueError('control_strength must be an integer or float.')
            if value < MIN_CONTROL_STRENGTH or value > MAX_CONTROL_STRENGTH:
                raise ValueError(
                    f'control_strength must be between {MIN_CONTROL_STRENGTH} and {MAX_CONTROL_STRENGTH}.'
                )
            return float(value)

        @field_validator('colors')
        def validate_colors(cls, value):
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError('colors must be a string with comma-separated colors.')
            return value

        @field_validator('similarity_strength')
        def validate_similarity_strength(cls, value):
            if not isinstance(value, (int, float)):
                raise ValueError('similarity_strength must be an integer or float.')
            if value < MIN_SIMILARITY_STRENGTH or value > MAX_SIMILARITY_STRENGTH:
                raise ValueError(
                    f'control_strength must be between {MIN_SIMILARITY_STRENGTH} and {MAX_SIMILARITY_STRENGTH}.'
                )
            return float(value)

    def __init__(self, model_id):
        self.model_id = model_id
        self.model_name = MODEL_ID_TO_MODEL_NAME[model_id]
        self.denied_regions = []  # Regions we tried, but didn't work.
        self.successful_region = None  # The region we successfully used.
        self.options = None  # Make options available throughout the class.

    def is_supported_in_region(self, region):
        """
        Discover if the given region supports the given model.

        :param region: An AWS Region name
        :return: True or False
        """
        bedrock = boto3.client('bedrock', region_name=region)
        response = bedrock.list_foundation_models(byOutputModality='IMAGE')
        models = [m['modelId'] for m in response['modelSummaries']]
        return self.model_id in models

    @cached_property
    def supported_regions(self):
        """
        Return a list of regions where this model is supported.
        We use a pool of concurrent threads to parallelize for speed.
        :return: A list of region names that support our model.
        """
        bedrock_regions = boto3.Session().get_available_regions('bedrock')

        model_regions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(bedrock_regions)) as executor:
            # Start the discovery operations and mark each future with its region
            future_to_region = {
                executor.submit(self.is_supported_in_region, region): region
                for region in bedrock_regions
            }
            for future in concurrent.futures.as_completed(future_to_region):
                region = future_to_region[future]
                if future.result():
                    model_regions.append(region)

        print_debug('Supported regions', model_regions)
        return model_regions

    @cached_property
    def region_to_location(self):
        """
        Return a region-to-location-names dict downloaded from the AWS pricing website.
        Return None if this information is not available.

        :return: a dict with AWS region codes as keys and AWS pricing region names as values.
        """
        r = requests.get(AWS_LOCATIONS_URL)
        if r.status_code != 200:
            return None

        result = {
            v['code']: v['name']
            for v in r.json().values()
            if v['type'] == 'AWS Region'
        }

        return result

    @cached_property
    def pricing(self):
        """
        Return a region-to-model-pricing dict downloaded from the AWS pricing website.
        Sort the list by price.
        Return None if this information is not available.

        :return: a dict with AWS region codes as keys and model pricing info dicts as values.
        """
        r = requests.get(AWS_BEDROCK_PRICING_URL)
        if r.status_code != 200:
            return None

        pricing = r.json()['regions']
        print_debug('Raw pricing', pricing)

        result = {}
        for region in self.supported_regions:
            pricing_region = self.region_to_location.get(region)
            if pricing_region not in pricing:  # Unlikely, but could happen. Discard if.
                self.denied_regions.append(region)
                continue

            for k, v in pricing[pricing_region].items():
                if not (
                    k.startswith('Amazon Bedrock On-demand Inference T2I') and
                    self.model_name in k
                ):
                    continue

                size, variant, name = k.split(' ', maxsplit=7)[5:]
                if result.get(region) is None:
                    result[region] = {}
                if result[region].get(size) is None:
                    result[region][size] = {}
                result[region][size][variant] = v['price']

        print_debug('Pricing data', result)
        return result

    @cached_property
    def regions_by_price(self):
        """
        Return a list of regions that support our model, sorted by price for this model.
        Prices come in output sizes and quality variants. We assume that price
        differences are consistent between sizes any variants within regions, so we can
        pick any size and variant and sort the regions by that price.
        If no pricing data is available, return the list of regions in any order.

        :return: A list of region names.
        """
        if not self.pricing:
            return self.supported_regions

        # Pick any size and variant, then sort the regions by it.
        regions = list(self.pricing.keys())
        size = list(self.pricing[regions[0]].keys())[0]  # First size.
        variant = list(self.pricing[regions[0]][size].keys())[0]  # First variant.

        return sorted(
            regions,
            key=lambda x: (
                self.pricing[x][size][variant],
                x  # if pricing is the same, sort alphabetically by region name.
            )
        )

    def load_and_preprocess_image(self, image_path):
        """
        Load and pre-process the image from the given image path for use with Titan Image Generator models.
        * Resize if needed.
        * Convert into a supported format if needed.
        * Do nothing if the image is already compatible.

        :param image_path: An image file path.
        :return: The base64 encoded image file bytes of the (preprocessed) image to include inside a Titan Image
                 Generator request parameter, as well as a (width, height) tuple of the result image size.
        """
        with open(image_path, "rb") as fp:
            img_bytes = fp.read()

        with Image.open(BytesIO(img_bytes)) as img:
            img_format = img.format
            width, height = img.size
            new_width = width
            new_height = height
            if min(width, height) < MIN_INPUT_IMAGE_LENGTH:
                if self.options.resize_image:
                    resize = MIN_INPUT_IMAGE_LENGTH / min(width, height)
                    new_width = int(width * resize + 0.5)
                    new_height = int(height * resize + 0.5)
                    if max(width, height) > MAX_INPUT_IMAGE_LENGTH:
                        raise ValueError(
                            'The minimum length of the input image is too short. Minimum image length: ' +
                            f'{MIN_INPUT_IMAGE_LENGTH}, request input image minimum length: {min(width, height)}.\n' +
                            'Unable to resize the image because then the maximum length would become too long.'
                        )
                else:
                    raise ValueError(
                        'The minimum length of the input image is too short. Minimum image length: ' +
                        f'{MIN_INPUT_IMAGE_LENGTH}, request input image minimum length: {min(width, height)}.'
                    )
            if width > MAX_INPUT_IMAGE_LENGTH or height > MAX_INPUT_IMAGE_LENGTH:
                if self.options.resize_image:
                    resize = MAX_INPUT_IMAGE_LENGTH / max(width, height)
                    new_width = int(width * resize)
                    new_height = int(height * resize)
                else:
                    raise ValueError(
                        'The maximum length of the input image is too long. Maximum image length: ' +
                        f'{MAX_INPUT_IMAGE_LENGTH}, request input image maximum length: {max(width, height)}.'
                    )
            if new_width != width or new_height != height:
                # Resize the image if needed
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Change format if necessary
            if (
                img_format.lower() in SUPPORTED_INPUT_IMAGE_FORMATS and
                img.size == (width, height)  # Original size, no resize needed
            ):
                return base64.b64encode(img_bytes).decode('utf8'), (width, height)

            # If not, re-export the image with the appropriate format
            with BytesIO() as buffer:
                img.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('utf8'), (new_width, new_height)

    def generate_text_image_params(self, prompt_text):
        """
        Generate the TEXT_IMAGE specific parameter block out of the given prompt text, using any options available from
        object’s options object.
        """
        text_to_image_params = {
            'text': prompt_text
        }
        if self.options.negative_prompt:
            text_to_image_params['negativeText'] = self.options.negative_prompt

        # Support for V2 condition images.
        if self.options.image:
            if 'image' not in MODEL_ID_TO_EXTRA_OPTIONS[self.model_id]:
                raise ValueError(f'Condition images are not supported with the {self.model_id} model.')

            image_b64, _ = self.load_and_preprocess_image(self.options.image)
            text_to_image_params['conditionImage'] = image_b64

            text_to_image_params['controlMode'] = self.options.control_mode
            text_to_image_params['controlStrength'] = self.options.control_strength

        return text_to_image_params

    def generate_painting_params(self, prompt_text):
        """
        Generate the shared portion of a Titan Image Generator parameter dict for INPAINTING and
        OUTPAINTING tasks.
        """
        if self.options.mask_image and self.options.mask_prompt:
            raise ValueError('Only one of mask_image or mask_prompt can be provided for IN/OUTPAINTING tasks.')
        if not self.options.mask_image and not self.options.mask_prompt:
            raise ValueError('Either mask_image or mask_prompt must be provided for IN/OUTPAINTING tasks.')
        if not self.options.image:
            raise ValueError('An input image is required for IN/OUTPAINTING tasks.')

        image_b64, image_size = self.load_and_preprocess_image(self.options.image)
        painting_params = {
            'image': image_b64,
        }

        if prompt_text:  # prompt is optional, in this case, the model will fill in the mask.
            painting_params['text'] = prompt_text

        if self.options.negative_prompt:
            painting_params['negativeText'] = self.options.negative_prompt

        if self.options.mask_prompt:
            painting_params['maskPrompt'] = self.options.mask_prompt

        if self.options.mask_image:
            mask_image_b64, mask_image_size = self.load_and_preprocess_image(self.options.mask_image)
            if mask_image_size != image_size:
                raise ValueError('The mask image must have the same size as the input image.')
            painting_params['maskImage'] = mask_image_b64

        return painting_params

    def generate_inpainting_params(self, prompt_text):
        """
        Generate a Titan Image Generator parameter dict for INPAINTING tasks.
        """
        return self.generate_painting_params(prompt_text)

    def generate_outpainting_params(self, prompt_text):
        """
        Generate a Titan Image Generator parameter dict for OUTPAINTING tasks.
        """
        out_painting_params = self.generate_painting_params(prompt_text)
        out_painting_params['outPaintingMode'] = self.options.outpainting_mode
        return out_painting_params

    def generate_image_variation_params(self, prompt_text):
        """
        Generate the IMAGE_VARIATION specific parameter block out of the given prompt text, using any options available
        from the object’s options object.
        """
        if not self.options.image and not self.options.images:
            raise ValueError(f'IMAGE_VARIATION_TASKS require at least one -o image or -o images option.')

        image_paths = []
        if self.options.image:
            image_paths.append(os.path.expanduser(self.options.image))

        if self.options.images:
            for i in self.options.images.split(','):
                image_paths.append(os.path.expanduser(i))

        if not image_paths:
            raise ValueError('The IMAGE_VARIATION task requires a minimum of 1 input image.')

        if len(image_paths) > MAX_INPUT_IMAGES_FOR_VARIATION:
            raise ValueError(
                f'The IMAGE_VARIATION task only supporta a maximum of {MAX_INPUT_IMAGES_FOR_VARIATION} images.'
            )

        images = []
        for i in image_paths:
            image_b64, _ = self.load_and_preprocess_image(i)
            images.append(image_b64)

        image_variation_params = {
            'text': prompt_text,
            'images': images,
            'similarityStrength': self.options.similarity_strength,
        }
        if self.options.negative_prompt:
            image_variation_params['negativeText'] = self.options.negative_prompt

        return image_variation_params

    @staticmethod
    def parse_colors(colors):
        """
        Parse the given colors string with comma-separated colors into a list of well-defined, 6-digit hex colors.
        Support human-friendly color names through the webcolors library.
        Also, perform some semantic checking of the color list based on the Titan Image Generator V2
        specs.
        """
        color_list = [s.strip() for s in colors.split(',')]

        if len(color_list) < MIN_COLORS:
            raise ValueError(f'colors must have at least one color.')
        if len(color_list) > MAX_COLORS:
            raise ValueError(f'colors cannot have more than {MAX_COLORS} colors.')

        return [
            webcolors.normalize_hex(color)
            if re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', color)
            else webcolors.name_to_hex(color)  # Raises ValueError if name is unknown.
            for color in color_list
        ]

    def generate_color_guided_generation_params(self, prompt_text):
        """
        Generate the COLOR_GUIDED_GENERATION specific parameter block out of the given options object.
        Assumes that any options are available from self.options.
        """
        if not self.options.colors:
            raise ValueError(
                'A set of colors to guide the image generation process with is required for COLOR_GUIDED_GENERATION ' +
                'tasks.'
            )

        color_guided_generation_params = {
            'text': prompt_text,
            'colors': self.parse_colors(self.options.colors)
        }
        if self.options.negative_prompt:
            color_guided_generation_params['negativeText'] = self.options.negative_prompt

        if self.options.image:
            image_b64, _ = self.load_and_preprocess_image(self.options.image)
            color_guided_generation_params['referenceImage'] = image_b64

        return color_guided_generation_params

    def generate_background_removal_params(self):
        """
        Generate the BACKGROUND_REMOVAL specific parameter block for background removal tasks.
        This task does not use any prompt text.
        Assumes that any options are available from self.options.
        """
        if self.options.image:
            image_b64, _ = self.load_and_preprocess_image(self.options.image)
            return {
                'image': image_b64
            }
        else:
            ValueError('An input image is required for BACKGROUND_REMOVAL tasks.')

    def generate_detect_generated_content_params(self):
        """
        Create a params dict (not just a sub-dict like in the other cases) for the Bedrock
        detect_generated_content() API (preview).

        See also: https://aws.amazon.com/de/blogs/aws/
        amazon-titan-image-generator-and-watermark-detection-api-are-now-available-in-amazon-bedrock/

        As of 2024-08-21, this feature is in preview and requires a special preview SDK which is not generally
        available yet. So we’ll leave this as is for now until it becomes more widely available.
        """
        with open(self.options.image, "rb") as image_file:
            input_image = image_file.read()

        return {
            "imageContent": {"bytes": input_image}
        }

    def prompt_to_titan_params(self, prompt):
        """
        Generate a Bedrock Titan image parameters dict based on the given prompt.
        :param prompt: A llm prompt object.
        :return: A dict of Bedrock Titan image parameters.
        """
        # Verify that we support the given task.
        task_type = prompt.options.task
        if task_type not in MODEL_ID_TO_TASK_TYPES[self.model_id]:
            raise ValueError(
                f'The {task_type} task type is not supported with the {self.model_id} model.'
            )

        # Build and validate the text prompt. Combine it with the system prompt if any.
        prompt_text = prompt.prompt
        if prompt.system:
            prompt_text = prompt.system + ' ' + prompt_text

        if not prompt_text.strip() and prompt.options.task not in TASKS_WITH_NO_PROMPT_NEEDED:
            raise ValueError('Prompt cannot be empty.')
        if len(prompt_text) > MAX_PROMPT_LENGTH:
            raise ValueError(
                "Prompt is too long. Maximum prompt length for Amazon Titan Image " +
                "Generator G1 models is 512 characters."
            )

        # Validate the combination of width and height.
        size = (prompt.options.width, prompt.options.height)
        if size not in IMAGE_SIZES:
            raise ValueError(
                f"Invalid image size {size}. Valid sizes are: {', '.join([str(i) for i in IMAGE_SIZES])}."
            )

        params = {
            'taskType': task_type,
            'imageGenerationConfig': {
                'width': size[0],
                'height': size[1],
                'numberOfImages': prompt.options.number_of_images,
                'cfgScale': prompt.options.cfg_scale,
                'seed': prompt.options.seed,
                'quality': prompt.options.quality
            }
        }

        if task_type == 'TEXT_IMAGE':
            params['textToImageParams'] = self.generate_text_image_params(prompt_text)
        elif task_type == 'INPAINTING':
            params['inPaintingParams'] = self.generate_inpainting_params(prompt_text)
        elif task_type == 'OUTPAINTING':
            params['outPaintingParams'] = self.generate_outpainting_params(prompt_text)
        elif task_type == 'IMAGE_VARIATION':
            params['imageVariationParams'] = self.generate_image_variation_params(prompt_text)
        elif task_type == 'COLOR_GUIDED_GENERATION':
            params['colorGuidedGenerationParams'] = self.generate_color_guided_generation_params(prompt_text)
        elif task_type == 'BACKGROUND_REMOVAL':
            params['backgroundRemovalParams'] = self.generate_background_removal_params()
        # elif task_type == 'DETECT_GENERATED_CONTENT':
        #     params = self.generate_detect_generated_content_params()
        else:
            raise ValueError(f'Invalid task type {task_type}.')

        print_debug('Titan params', params)

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
        if not base:
            base = "result"

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
        See: https://stackoverflow.com/questions/434597/open-document-with-default-os-application-in-python-both-
             in-windows-and-mac-os

        :param path: The path to a file to be opened.
        :return: None
        """
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', path))
        elif platform.system() == 'Windows':  # Windows
            # os.startfile(path) # Replaced by Amazon Q Developer with the below line to improve security.
            subprocess.run([path], shell=False, check=True, capture_output=True, text=True)
        else:  # linux variants
            subprocess.call(('xdg-open', path))

    @property
    def region(self):
        """
        Return the next best supported region for this model.
        If we know a region was successfully used, return that.
        If not, find the next best region choice based on the AWS_DEFAULT_REGION environment
        variable and a price-sorted list of regions.
        Also, return a reason for why it was chosen.
        Return None if no more regions are available for trying out.

        :return: A region name, reason string tuple.
        """
        if self.successful_region:
            return self.successful_region, 'from previous successful region'

        # Did the user set the region option?
        if self.options and self.options.region:
            region = self.options.region
            if self.is_supported_in_region(region) and region not in self.denied_regions:
                return region, 'from -o region option'
            raise ValueError(f'{self.options.region} not supported or not accessible.')

        # Is the AWS_DEFAULT_REGION environment variable set?
        region = os.environ.get('AWS_DEFAULT_REGION')
        if region and self.is_supported_in_region(region) and region not in self.denied_regions:
            return region, 'from AWS_DEFAULT_REGION environment variable'

        # If AWS_DEFAULT_REGION didn't work and auto_region is true, choose the lowest cost one.
        if self.options and self.options.auto_region:
            for region in self.regions_by_price:
                if region not in self.denied_regions:
                    return region, 'lowest cost region supporting this model'

        raise ValueError('AWS_DEFAULT_REGION not supported or accessible, no more regions left to try.')

    def process_response_images(self, response_body, prompt):
        """
        Process all images contained in the given response body. Use the given prompt to generate
        appropriate file names. yield processed file names while processing.
        Remove them from the body after processing to avoid bulk when logging the rest.
        """
        for i, image_data in enumerate(response_body.get("images")):
            image_bytes = base64.b64decode(image_data)
            image_filename = self.prompt_to_filename(prompt)
            with open(image_filename, "wb") as fp:
                fp.write(image_bytes)

            if prompt.options.auto_open:
                self.open_file(image_filename)

            response_body['images'][i] = f'<image data {i + 1}>'
            yield image_filename

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
        self.options = prompt.options
        params = self.prompt_to_titan_params(prompt)

        # There's no way to discover whether we have a subscription for a model in a specific
        # region, other than trying.
        bedrock_response = None

        region, region_reason = self.region
        while region and not bedrock_response:
            if stream:
                yield f'Calling {self.model_name} on Bedrock in the {region} region ({region_reason}).\n'

            bedrock = boto3.client('bedrock-runtime', region_name=region)
            print_debug('Prompt body', params)

            try:
                bedrock_response = bedrock.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(params)
                )

                self.successful_region = region
                break
            except bedrock.exceptions.AccessDeniedException:
                if stream:
                    yield f'Access denied. Please consider requesting access to this model in the {region} region.\n'

                self.denied_regions.append(region)
                region, region_reason = self.region

        response_body = json.loads(bedrock_response.get("body").read())
        image_filenames = []
        # removes images from response_body.
        for image_filename in self.process_response_images(response_body, prompt):
            image_filenames.append(image_filename)
            if stream:
                yield f"Image written to: {image_filename}\n"

        file_names = [self.prompt_to_filename(prompt)]

        # Log the rest of the response body in case there's more useful information in it.
        response.response_json = json.dumps(response_body)

        if not stream:
            if len(file_names) == 1:
                yield (
                        f'Image generated in the {region} Region ({region_reason}) ' +
                        f'and written to: {file_names[0]}.\n'
                )
            else:
                yield (
                        f'Images generated in the {region} region ({region_reason}.\n' +
                        'Images written to:\n' +
                        f'{"\n    ".join(file_names)}\n'
                )
