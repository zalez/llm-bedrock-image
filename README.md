# llm-bedrock-image

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sblakey/llm-bedrock-anthropic/blob/main/LICENSE)

Plugin for Simon Willison’s [LLM](https://llm.datasette.io) CLI utility, adding support for Amazon Titan Image Generator
G1 models on Amazon Bedrock.

## Installation

This installation assumes you have a working installation of Simon Willison’s LLM tool. If not, see:
[LLM Setup](https://llm.datasette.io/en/stable/setup.html)

Install from GitHub:

```bash
git clone https://github.com/zalez/llm-bedrock-image-titan.git
cd <directory you cloned into>
llm install -e .
```

At some point, this will by published onto PyPi, after which you should be able to just say:

```bash
llm install llm-bedrock-image
```

## Configuration

This plugin uses the same standard AWS configuration environment variables as other AWS CLI and SDK tools.
It is based on Boto3, the AWS SDK for Python. Check the
[Boto3 Documentation on Configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration)
on how to set up your AWS environment.

Also, you need to have successfully requested access to one of the Amazon Titan Image Generator G1 models on Amazon
Bedrock. See
[Manage access to Amazon Bedrock foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
from the Amazon Bedrock Documentation on how to do this.

## Quickstart

The easiest way to use this is:

```bash
llm -m bti "A parrot."
```

If successful, this will generate an image of a parrot and save it as "A_parrot.png" at your current file system path.

The model identifier "bti" is shorthand for "Bedrock Titan Image", the model we’re using here. "A parrot." is the prompt
we use to generate the image. Note that while LLM assumes we’re having a conversation with an LLM here, we’re merely
generating images. However, we do handle useful output, like the resulting filename, as if we had received a
conversation response from a "chat" LLM. This conversation is logged by LLM like any other LLM conversation.

