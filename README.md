# llm-bedrock-titan-image

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/zalez/llm-bedrock-titan-image/blob/main/LICENSE)

Plugin for Simon Willison’s [LLM](https://llm.datasette.io) CLI utility, adding support for Amazon Titan Image Generator
G1 models on Amazon Bedrock.

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
[LLM Setup](https://llm.datasette.io/en/stable/setup.html)

Install from GitHub:

```bash
git clone https://github.com/zalez/llm-bedrock-titan-image.git
cd <directory you cloned into>
llm install -e .
```

At some point, this will be published onto PyPi, after which you should be able to just say:

```bash
llm install llm-bedrock-titan-image
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

If everything is set up correctly, this will generate an image of a parrot and save it as "A_parrot.png" in your
current file system path:

![A parrot.](A_parrot.png)

If a file with the same name exists already (likely, since this repository has one), this plugin will choose an unused
version of the filename, like ```A_parrot_1.png```.

The model identifier "ati" is shorthand for "amazon.titan-image-generator-v2:0", the model we specified here.
"A parrot." is the prompt used to generate the image. Note that while LLM assumes a conversation with an LLM, we’re
merely generating images here. However, we do handle useful output, like the resulting filename, as if we
had received a conversation response from a "chat" LLM. This is logged by LLM like any other LLM conversation.

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

## Task types

At this time, only the basic Text-to-Image (```TEXT_IMAGE```) task type is supported. Over time, more task types may
be added.

## Options

### -o region REGION

The AWS region to use for this model. Overrides the AWS_DEFAULT_REGION environment, if set. If none is configured and
auto_region (see below) is true, this plugin will automatically choose a supported and enabled region.

If this option is set, the auto_region option is ignored.

Valid values: Any AWS Region where the Amazon Bedrock service is available. As of 2024-08-16, these are: 
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

Valid values: ```true```, ```false```, ```on```, ```off```, ```0```, ```1```.

Default: ```true```

Example:
```bash
llm -m ati "A parrot." -o auto_region off
```

### -o auto_open [true|false|on|off|0|1]

Automatically opens the image after generation using the operating system’s default image viewer. This has been
successfully tested on macOS, but should work on Windows and Linux, too.

Valid values: ```true```, ```false```, ```on```, ```off```, ```0```, ```1```.

Default: ```true```

Example:
```bash
llm -m ati "A parrot." -o auto_open off
```

### -o negative_prompt TEXT

Specify a second prompt describing what to avoid generating in the image.

Valid values: Any string describing what not to generate. Don’t use "no" or other negations here, specify what to
avoid in positive words.

Default: None

Example:
```bash
llm -m ati "A mouse." -o negative_prompt "computer, electronics, device"
```

### -o number_of_images INT

The number of images to generate.

Valid values: integers from 1 through 5.

Default: 1

Example:
```bash
llm -m ati "A parrot." -o number_of_images 3
```

### -o cfg_scale FLOAT

Specifies how strongly the generated image should adhere to the prompt. Use a lower value to introduce more randomness
in the generation

Valid values: Any floating-point number between 1.1 and 10.0.

Default: 8.0

Example:
```bash
llm -m ati "A parrot." -o cfg_scale 9.0
```

### -o height INT

The height of the image to generate in pixels. See the
[Amazon Titan Image Generator G1 documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html)
for valid width/height combinations.

Valid values: one of 320, 384, 448, 512, 576, 640, 768, 704, 768, 896, 1024, 1152, 1280, 1408.

Default: 1024

Example:
```bash
llm -m ati "A parrot." -o width 768 -o height 1152
```

### -o width INT

The width of the image to generate in pixels. See the
[Amazon Titan Image Generator G1 documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html)
for valid width/height combinations.

Valid values: one of 320, 384, 448, 512, 576, 640, 768, 704, 768, 896, 1024, 1152, 1173, 1280, 1408.

Default: 1024

Example:
```bash
llm -m ati "A parrot." -o width 768 -o height 1152
```

### -o seed INT

Use this to control and reproduce results. Determines the initial noise setting. Use the same seed and the same
settings as a previous run to allow inference to create a similar image.

Valid values: any integer between 0 and 2147483646.

Default: 0

Example:
```bash
llm -m ati "A dolphin." -o seed 42
```

### -o quality [standard|premium]

The quality to generate image with. Use "standard" for drafting and "premium" to get finder details.

Valid values: ```standard```, ```premium```

Default: ```standard```

Example:
```bash
llm -m ati "A parrot." -o quality premium
```