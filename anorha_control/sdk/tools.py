"""
Tool definitions for LLM function calling (OpenAI, Anthropic format).
"""
from typing import List, Dict, Any

# OpenAI tools format (for chat.completions.create(tools=...))
TOOLS_OPENAI: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "browser_goto",
            "description": "Navigate the browser to a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to navigate to"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click on an element. Use semantic description, e.g. 'login button', 'search box', 'submit'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Semantic description of the element to click"},
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type text. Optionally focus an element first by target description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                    "target": {"type": "string", "description": "Optional. Element to focus first (e.g. 'search box')"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scroll the page up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"], "description": "Scroll direction"},
                    "amount": {"type": "integer", "description": "Scroll amount in pixels", "default": 300},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "Capture a screenshot. Returns base64 image for vision LLMs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_press_key",
            "description": "Press a key (e.g. Enter, Tab, Escape).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key to press (Enter, Tab, Escape, etc.)"},
                },
                "required": ["key"],
            },
        },
    },
]

# Anthropic tools format (similar structure)
TOOLS_ANTHROPIC: List[Dict[str, Any]] = [
    {
        "name": "browser_goto",
        "description": "Navigate the browser to a URL.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to navigate to"}},
            "required": ["url"],
        },
    },
    {
        "name": "browser_click",
        "description": "Click on an element. Use semantic description, e.g. 'login button', 'search box'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Semantic description of the element to click"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "browser_type",
        "description": "Type text. Optionally focus an element first by target description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"},
                "target": {"type": "string", "description": "Optional. Element to focus first"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "browser_scroll",
        "description": "Scroll the page up or down.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"]},
                "amount": {"type": "integer", "default": 300},
            },
            "required": ["direction"],
        },
    },
    {
        "name": "browser_screenshot",
        "description": "Capture a screenshot. Returns base64 image for vision LLMs.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "browser_press_key",
        "description": "Press a key (e.g. Enter, Tab, Escape).",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Key to press"}},
            "required": ["key"],
        },
    },
]
