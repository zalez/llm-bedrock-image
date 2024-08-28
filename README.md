# llm-bedrock-titan-image

[![PyPI](https://img.shields.io/pypi/v/llm-bedrock-titan-image.svg)](https://pypi.org/project/llm-bedrock-titan-image/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/zalez/llm-bedrock-titan-image/blob/main/LICENSE)

A plugin for Simon Willison’s [LLM](https://llm.datasette.io) CLI utility, adding support for Amazon Titan Image Generator 
G1 models V1 and V2 on Amazon Bedrock.

## Disclaimer

This plugin is unofficial and in no way affiliated with or endorsed by Amazon Web Services (AWS). This is merely a hobby
project, provided "as-is" with no further commitment or promises at this time.

That said, feedback, feature suggestions, and pull requests are welcome.

## Amazon Titan Image Generator G1 models

Amazon Titan Image Generator G1 is a generative image model that allows users to generate realistic, studio-quality
images using text prompts.

Learn more about this model from the [product page](https://aws.amazon.com/bedrock/titan/#Amazon_Titan_Image_Generator), its [documentation page](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-image-models.html), its
[parameter documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html), and
its [prompt engineering best practices documentation](https://d2eo22ngex1n9g.cloudfront.net/Documentation/User+Guides/Titan/Amazon+Titan+Image+Generator+Prompt+Engineering+Guidelines.pdf).

## Cost

Using these models comes at some cost, billed to your AWS account. Please refer to the
[Amazon Bedrock pricing page for Titan multi-modal models](https://aws.amazon.com/bedrock/pricing/#Amazon) for details.

## Security

This plugin uses the standard AWS SDK for Python, Boto3, to access Amazon Bedrock, through standard methods. Please
refer to the [Boto3 security documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/security.html)
and the [AWS Security documentation for Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/security.html)
to learn more about security topics related to the underlying SDK and Amazon Bedrock.

This plugin assumes that no security relevant or personally identified information is used in prompts in any way and
that you are ok with prompt text to be transferred globally in an encrypted way to any AWS Region endpoint world-wide
for execution. By setting the ```AWS_DEFAULT_REGION``` environment variable, and setting the ```auto_region``` option to
```off```, you can ensure that the plugin only uses your chosen region to execute your prompt.

## Installation

This plugin assumes you have a working installation of Simon Willison’s LLM tool. If not, see:
[LLM Setup](https://llm.datasette.io/en/stable/setup.html).

The plugin is available from PyPi as: [llm-bedrock-titan-image](https://pypi.org/project/llm-bedrock-titan-image/), so
it can be easily installed from ```llm``` itself:

```bash
llm install llm-bedrock-titan-image
```

Install from GitHub is also possible, but only necessary if the main commit hasn't been published yet on PyPi, or if
you have forked/modified your own version:

```bash
git clone https://github.com/zalez/llm-bedrock-titan-image.git
cd <directory you cloned into>
llm install -e .
```

## Configuration

This plugin uses the same standard AWS configuration environment variables as other AWS CLI and SDK tools.
It is based on Boto3, the AWS SDK for Python. Check the
[Boto3 Documentation on Configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration)
on how to set up your AWS environment.

Also, you need to have successfully requested access to one of the Amazon Titan Image Generator G1 models on Amazon
Bedrock using the AWS Console. See
[Manage access to Amazon Bedrock foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
from the Amazon Bedrock Documentation on how to do this.

## Quickstart

An easy way to use this plugin is:

```bash
llm -m ati "A parrot."
```

If everything is set up correctly, this will generate an image of a parrot, save it as "A_parrot.png" in your
current file system path, and open it with your operating system’s default image display app:

![A parrot.](https://raw.githubusercontent.com/zalez/llm-bedrock-titan-image/main/A_parrot.png)

If a file with the same name exists already (likely, since this repository has one), this plugin will choose an unused
version of the filename, like ```A_parrot_1.png```.

The model identifier "ati" is shorthand for "amazon.titan-image-generator-v2:0", the model we specified here.
"A parrot." is the prompt (up to 512 characters) used to generate the image. Note that while LLM assumes a conversation
with an LLM, we’re merely generating images here. However, we do handle useful output, like the resulting filename, as
if we had received a conversation response from a "chat" LLM. This is logged by LLM like any other LLM conversation.

When crafting prompts, check out the
[Amazon Titan Image Generator Prompt Engineering Best Practices documentation](https://d2eo22ngex1n9g.cloudfront.net/Documentation/User+Guides/Titan/Amazon+Titan+Image+Generator+Prompt+Engineering+Guidelines.pdf)
for helpful guidelines and examples.

## Available models

Both Amazon Titan Image Generator G1 models are available: V1 and V2. Learn more about them here:

https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

Use their official model IDs or the following aliases to request them:

### Amazon Titan Image Generator G1 V1

Model-ID: ```amazon.titan-image-generator-v1```, aliases: ```amazon-titan-image-v1```, ```ati1```.

### Amazon Titan Image Generator G1 V2

Model-ID: ```amazon.titan-image-generator-v2:0```, aliases: ```amazon-titan-image-v2```, ```amazon-titan-image```,
```ati2```, ```ati```.

Includes support for the control image feature.

Example for using the older V1 model:

```bash
llm -m amazon-titan-image-v1 "A parrot."
```

## Task types

This plugin currently supports the following task types:

### TEXT_IMAGE

Generate an image using a text prompt.

Optionally (V2 only), Provide an additional input conditioning image along with a text prompt to generate an image that
follows the layout and composition of the conditioning image.

Examples:

```bash
llm -m ati -o task TEXT_IMAGE "A parrot."
llm -m ati -o task TEXT_IMAGE "A parrot." -o image /path/to/condition_image.jpg
```

### INPAINTING

Modify an image by changing the inside of a mask to match the surrounding background.

The mask can be either generated by the model using a mask prompt through ```-o mask_prompt``` or supplied as a separate
mask image using ```-o mask image PATH/TO/IMAGE.png```. Mask images can only have two colors: Black (RGB: 0, 0, 0) for
the mask and White (RGB: 255, 255, 255) for everything else.

Use the regular prompt to describe what you want to generate inside the mask. If an empty prompt (```"""```) is given,
the model will try to fill the mask with a continuation of the background.

Examples:

```bash
llm -m ati -o task INPAINTING -o image A_parrot.png -o mask_prompt 'Parrot' 'Replace the parrot with an iguana.'
llm -m ati -o task INPAINTING -o image Vacation.jpg -o mask_image Vacation_clouds_mask.png ""
```

### OUTPAINTING

Modify an image by seamlessly extending the region defined by the mask.

The mask can be either generated by the model using a mask prompt through ```-o mask_prompt``` or supplied as a separate
mask image using ```-o mask image PATH/TO/IMAGE.png```. Mask images can only have two colors: Black (RGB: 0, 0, 0) for
the mask and White (RGB: 255, 255, 255) for everything else.

Use the regular prompt (required) to describe what you want to generate outside of the mask.

Examples:

```bash
llm -m ati -o task OUTPAINTING -o mask_prompt 'people' -o image vacation.jpeg 'in a jungle.'
llm -m ati -o task OUTPAINTING -o mask_image vacation_mask.png -o image vacation.jpeg 'in a jungle.'
```

### IMAGE_VARIATION

Image variation allow you to create variations of your original image(s) based on the parameter values.

Describe in the prompt what to keep and what to modify in the image(s). Either supply one image using the
```-o image``` option or multiple, comma-separated image paths using the ```-o images``` option.

Examples:

```bash
llm -m ati -o task variation -o image photo.jpg 'Make it look more modern.'
llm -m ati -o task variation -o images photo_1.jpg,photo_2.jpg 'Cartoon style.'
```

### COLOR_GUIDED_GENERATION

(V2 only.)

Provide a list of hex color codes or CSS3 color names along with a text prompt to generate an image that follows the
color palette. Also supports adding an image to guide the colory styling.

Examples:
```bash
llm -m ati -o task COLOR -o colors '#ff8800,yellow,blue' 'A parrot.'
llm -m ati -o task COLOR -o colors "yellow, blue" -o image design_example.jpg
```

### BACKGROUND_REMOVAL

(V2 only.)

Detect and remove the background of the given image. No prompt is required, however, llm will insist on a prompt and
wait for input if no prompt is supplied. Therefore, adding an empty prompt like "" is necessary.

Example:
```bash
llm -m ati -o task background -o image 'A_parrot.png' ""
```

## Common options

The following options are available for all tasks:

### -o region REGION

The AWS region to use for this model. Overrides the AWS_DEFAULT_REGION environment, if set. If none is configured and
auto_region (see below) is true, this plugin will automatically choose a supported and enabled region.

If this option is set, the auto_region option is ignored.

Valid regions are any AWS Region where the Amazon Bedrock service is available. As of 2024-08-19, these are: 
```ap-northeast-1```, ```ap-south-1```, ```ap-southeast-1```, ```ap-southeast-2```, ```ca-central-1```,
```eu-central-1```, ```eu-west-1```, ```eu-west-2```, ```eu-west-3```, ```sa-east-1```, ```us-east-1```,
```us-west-2```.

Default: None

Example:
```bash
llm -m ati "A parrot." -o region us-west-2
```

### -o auto_region [true|false|on|off|0|1]

Choose a supported AWS Region automatically if no default Region is configured in the AWS_DEFAULT_REGION environment
variable or the model isn’t supported or requested there.

If the region specified in AWS_DEFAULT_REGION does not support the chosen model, or if access has not been granted for
this model in the specified region, the plugin will try out all supported regions in lowest-price-first order, until
it finds a region that works.

**Data privacy note**: since image generator use-cases on non-fine-tuned models typically don’t involve personally
identifiable data, this plugin assumes that sending your given prompt to any AWS region is acceptable, when trying to
find one where you have requested access to. If you want to control the region this tool uses the model in, set your
chosen Region up in the ```ÀWS_DEFAULT_REGION``` environment variable and set this option to "```false```", "```off```",
or "```0```".

Default: ```true```

Example:
```bash
llm -m ati "A parrot." -o auto_region off
```

### -o task [TEXT_IMAGE|INPAINTING|OUTPAINTING|IMAGE_VARIATION|COLOR_GUIDED_GENERATION|BACKGROUND_REMOVAL]

The task type to execute. See *Task types* above for available task types and what they do.
Any upper/lower/mixed case string is allowed that matches exactly one of the available task types unambiguously.

Default: ```TEXT_IMAGE```

Example:
```bash
llm -m ati "A parrot." -o task image
```

### -o auto_open [true|false|on|off|0|1]

Automatically opens the image after generation using the operating system’s default image viewer. This has been
successfully tested on macOS, but should work on Windows and Linux, too.

Default: ```true```

Example:
```bash
llm -m ati "A parrot." -o auto_open off
```

### -o number_of_images INT

The number of images to generate.

Valid values: 1 - 5

Default: 1

Example:
```bash
llm -m ati "A parrot." -o number_of_images 3
```

### -o cfg_scale FLOAT

Specifies how strongly the generated image should adhere to the prompt. Use a lower value to introduce more randomness
in the generation

Valid values: floating-point numbers between 1.1 and 10.0.

Default: 8.0

Example:
```bash
llm -m ati "A parrot." -o cfg_scale 9.0
```

### -o height INT

The height of the image to generate in pixels. See the
[Amazon Titan Image Generator G1 documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html)
for valid width/height combinations.

Valid values include one of: 320, 384, 448, 512, 576, 640, 768, 704, 768, 896, 1024, 1152, 1280, 1408.

Default: 1024

Example:
```bash
llm -m ati "A parrot." -o width 768 -o height 1152
```

### -o width INT

The width of the image to generate in pixels. See the
[Amazon Titan Image Generator G1 documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html)
for valid width/height combinations.

Valid values include one of: 320, 384, 448, 512, 576, 640, 768, 704, 768, 896, 1024, 1152, 1173, 1280, 1408.

Default: 1024

Example:
```bash
llm -m ati "A parrot." -o width 768 -o height 1152
```

### -o seed INT

Use this to control and reproduce results. Determines the initial noise setting. Use the same seed and the same
settings as a previous run to allow inference to create a similar image.

Valid values: 0 - 2147483646

Default: 0

Example:
```bash
llm -m ati "A dolphin." -o seed 42
```

### -o quality [STANDARD|PREMIUM]

The quality to generate image with. Case-insensitive. Use "standard" for drafting and "premium" to get finer details.

Default: ```STANDARD```

Example:
```bash
llm -m ati "A parrot." -o quality premium
```

### -o image PATH/TO/IMAGE.JPG

Available with: ```TEXT_IMAGE```, ```INPAINTING```, ```OUTPAINTING```, ```COLOR_GUIDED_GENERATION```

(Titan Image Generator V2 only.)
The file system path to an image to use for guiding certain aspects of certain tasks:

* For ```TEXT_IMAGE```, the image is used to guide the overall image generation process.
* For ```INPAINTING``` and ```OUTPAINTING```, the image to modify.
* For ```IMAGE_VARIATION```, the image to create variation(s) for.
* For ```COLOR_GUIDED_GENERATION```, the image supplies a color theme for the model to use.
* For ```BACKGROUND REMOVAL```, this is the image to remove background from.

While the image formats supported by the models include ```.jpg``` and ```.png```, this plugin will automatically
transcode other formats into ```.png``` if necessary.

Images should have a minimum of 256 pixels on either side and a maximum of 1408 pixels on the longer side. The
```-o resize_image``` (default: on) can be used to automatically adjust the image size to fit the larges supported one.

Default: None

Example:

```bash
llm -m ati "A parrot." -o image /path/to/reference_image.jpg
llm -m ati "A parrot." -o task COLOR -o colors "yellow, blue" -o image design_example.jpg
llm -m ati -o task BACKGROUND -o image A_parrot.jpg ""  # NOTE: An empty prompt is required here, otherwise llm will wait for input.
```

### -o resize_image [true|false|on|off|0|1]

Resize the image(s) if it/they doesn't/don't conform to the model's supported image sizes.

Default: True
Example:

```bash
llm -m ati "A parrot." -o resize_image off -o image /path/to/reference_image.jpg
```

## Task-specific options

The following options are only available with some tasks types:

### -o negative_prompt TEXT

Available with: ```TEXT_IMAGE```, ```INPAINTING```, ```OUTPAINTING```, ```COLOR_GUIDED_GENERATION```.

Specify a second prompt describing what to avoid generating in the image.

Valid text: Any string up to 512 characters describing what not to generate. Specify what to avoid in positive words
without using "no" or other negations.

Default: None

Example:
```bash
llm -m ati "A mouse." -o negative_prompt "computer, electronics, device"
```

### -o control_mode [CANNY_EDGE|SEGMENTATION]

Available with: ```TEXT_IMAGE```.

The type of image conditioning to use with ```TEXT_IMAGE``` tasks (V2 only) when supplying a condition image:

* ```CANNY_EDGE```: uses edge detection on the condition image, while 
* ```SEGMENTATION```: uses semantic segmentation.

Assumes that an image has been provided through ```-o image```

See the [Amazon Titan Image Generator v2](https://aws.amazon.com/de/blogs/aws/amazon-titan-image-generator-v2-is-now-available-in-amazon-bedrock/) blog post for details.

Default: ```CANNY_EDGE```

Example:

```bash
llm -m ati "A cartoon parrot." -o image A_parrot.png -o control_mode SEGMENTATION
```

### -o control_strength FLOAT|INT

Available with: ```TEXT_IMAGE```.

A floating-point or integer value between 0.0 and 1.0 indicating how similar the layout and composition of the
generated image should be to the supplied image for ```TEXT_IMAGE``` tasks which add a condition image. Higher values
make the output more constrained to the condition image.

Valid values include integers or floating-point numbers between 0.0 and 1.0.

Default: 0.7

Example:

```bash
llm -m ati "A cartoon parrot." -o image A_parrot.png -o control_strength 0.9
```

### -o mask_prompt

Available with: ```INPAINTING```, ```OUTPAINTING```.

A text prompt that defines the mask. This mask is then used to define the area to change (```INPAINTING```) or the
area to preserve while generating content outside of it (```OUTPAINTING```)

Default: None

Example: 
```bash
llm -m ati -o task INPAINTING -o image A_parrot.png -o mask_prompt 'Parrot' 'Replace the parrot with an iguana.'
```

### -o mask_image

Available with: ```INPAINTING```, ```OUTPAINTING```.

A mask image that defines the area to change for ```INPAINTING``` tasks or the area to keep for ```OUTPAINTING```
tasks. Mask images can only have two colors: Black (RGB: 0, 0, 0) for the mask and White (RGB: 255, 255, 255) for
everything else.

Example:

```bash
llm -m ati -o task INPAINTING -o image Vacation.jpg -o mask_image Vacation_clouds_mask.png ""
```

### -o return_mask

Available with: ```INPAINTING```, ```OUTPAINTING```.

Also return the mask image that was used during an ```INPAINTING``` or ```OUTPAINTING``` task.
Useful for re-using the mask either with the model or an image editor.

Example:

```bash
llm -m ati -o task OUTPAINTING -o mask_prompt 'A parrot' -o image A_parrot.png -o return_mask true 'in a jungle.'
```

### -o images

Available with: ```IMAGE_VARIATION```.

Works like ```-o image```, only that it allows up to 5 images, with comma-separated paths.

Default: None

Example:
```bash
llm -m ati -o task variation -o images A_parrot.png,Another_parrot.png 'Cartoon style.'
```

### -o colors COLOR_LIST

Available with: ```COLOR_GUIDED_GENERATION```.

A comma-separated list of  1 - 10 hex color values or CSS3 color names for conditioning the colors used to generate
the image. Only supported with the ```COLOR_GUIDED_GENERATION``` task type, and required for this task.

Valid values: A string of comma-separated hex color values or CSS3 color names.

Default: None

Example:

```bash
llm -m ati -o task COLOR -o colors '#ff8800,yellow,blue' 'A parrot.'
```
