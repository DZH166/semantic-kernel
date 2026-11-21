# Copyright (c) Microsoft. All rights reserved.

import pytest

from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.function_result import FunctionResult
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_function_from_method import KernelFunctionFromMethod
from semantic_kernel.kernel import Kernel


def _chunk(choice_index: int, text: str) -> StreamingChatMessageContent:
    return StreamingChatMessageContent(role=AuthorRole.ASSISTANT, content=text, choice_index=choice_index)


def _stream_function(chunks_per_step: bool) -> KernelFunctionFromMethod:
    """A native streaming function that emits chunks for two choices out of order.

    When chunks_per_step is True, each yield is a list of contents (the shape used by
    the chat completion connectors); otherwise each yield is a single content object
    (the shape produced by native stream methods yielding individual contents).
    """

    @kernel_function(name="stream_two_choices", description="Streams chunks for two choices out of order.")
    async def stream_two_choices():
        steps = [_chunk(1, "B1 "), _chunk(0, "A1 "), _chunk(1, "B2"), _chunk(0, "A2")]
        for chunk in steps:
            yield [chunk] if chunks_per_step else chunk

    return KernelFunctionFromMethod(method=stream_two_choices, plugin_name="test")


def _in_order_stream_function() -> KernelFunctionFromMethod:
    """A native streaming function that emits list-shaped chunks for two choices in order."""

    @kernel_function(name="stream_two_choices", description="Streams list-shaped chunks for two choices in order.")
    async def stream_two_choices():
        yield [_chunk(0, "A1 "), _chunk(1, "B1 ")]
        yield [_chunk(0, "A2"), _chunk(1, "B2")]

    return KernelFunctionFromMethod(method=stream_two_choices, plugin_name="test")


@pytest.mark.asyncio
async def test_invoke_stream_return_function_results_merges_out_of_order_single_chunks():
    """Single content objects yielded by a native stream method must be merged by choice_index."""
    kernel = Kernel()
    function = kernel.add_function(plugin_name="test", function=_stream_function(chunks_per_step=False))

    collected = []
    async for message in kernel.invoke_stream(function=function, return_function_results=True):
        collected.append(message)

    final_results = [message for message in collected if isinstance(message, FunctionResult)]
    assert len(final_results) == 1, "one merged FunctionResult should be yielded"

    merged = final_results[0].value
    assert len(merged) == 2, "both choices should be present in the merged result"
    assert merged[0].choice_index == 0
    assert merged[0].content == "A1 A2"
    assert merged[1].choice_index == 1
    assert merged[1].content == "B1 B2"


@pytest.mark.asyncio
async def test_invoke_stream_return_function_results_merges_out_of_order_list_chunks():
    """Out-of-order choice indexes inside connector-style list chunks must not be
    merged into the wrong choice (today this raises ContentAdditionException)."""
    kernel = Kernel()
    function = kernel.add_function(plugin_name="test", function=_stream_function(chunks_per_step=True))

    collected = []
    async for message in kernel.invoke_stream(function=function, return_function_results=True):
        collected.append(message)

    final_results = [message for message in collected if isinstance(message, FunctionResult)]
    assert len(final_results) == 1, "one merged FunctionResult should be yielded"

    merged = final_results[0].value
    assert len(merged) == 2, "both choices should be present in the merged result"
    assert merged[0].choice_index == 0
    assert merged[0].content == "A1 A2"
    assert merged[1].choice_index == 1
    assert merged[1].content == "B1 B2"


@pytest.mark.asyncio
async def test_invoke_stream_return_function_results_merges_in_order_list_chunks_unchanged():
    """Pin the already-working connector path: list chunks with ascending choice indexes."""
    kernel = Kernel()
    function = kernel.add_function(plugin_name="test", function=_in_order_stream_function())

    collected = []
    async for message in kernel.invoke_stream(function=function, return_function_results=True):
        collected.append(message)

    final_results = [message for message in collected if isinstance(message, FunctionResult)]
    assert len(final_results) == 1

    merged = final_results[0].value
    assert len(merged) == 2
    assert merged[0].choice_index == 0
    assert merged[0].content == "A1 A2"
    assert merged[1].choice_index == 1
    assert merged[1].content == "B1 B2"


@pytest.mark.asyncio
async def test_invoke_stream_yields_chunks_in_arrival_order_without_merge():
    """Sanity: without return_function_results, every chunk is yielded as it arrives."""
    kernel = Kernel()
    function = kernel.add_function(plugin_name="test", function=_stream_function(chunks_per_step=False))

    collected = []
    async for message in kernel.invoke_stream(function=function):
        collected.append(message)

    assert [message.choice_index for message in collected] == [1, 0, 1, 0]
