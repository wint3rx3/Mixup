# Structured outputs

## **What is structured outputs?**

Structured outputs enable users to extract and organize information in a standardized format by generating a JSON string based on a provided JSON schema. This feature ensures that extracted data is both machine-readable and easily integrable into various applications.

## **What models does Upstage provide?**

The table below lists models currently available as an API. Upstage provides stable *aliases* that point to specific model versions, allowing you to integrate once and automatically benefit from future updates. We recommend using aliases instead of hardcoding model names, as models can be frequently updated.

| **Alias** | **Currently points to** | **RPM / TPM [(Learn more)](https://console.upstage.ai/docs/guides/rate-limits)** |
| --- | --- | --- |
| solar-pro | [solar-pro-250422](https://console.upstage.ai/docs/models#solar-pro-250422) | 100 / 100,000 |
| solar-mini | [solar-mini-250422](https://console.upstage.ai/docs/models#solar-mini-250422) | 100 / 100,000 |

## **Defining schemas**

- Structured outputs supports a subset of the [JSON Schema](https://json-schema.org/overview/what-is-jsonschema) specification. The supported types are: `string`, `number`, `boolean`, `integer`, `object`, and `array`.
- All fields must be explicitly listed in the `required` array.
- Objects are limited to a maximum nesting depth of 3 levels.
- The `strict` property must always be set to `true`.
- The value of additionalProperties must always be `false`.
- Definitions for subschemas are not supported.
- Recursive schemas are not supported.

## **Examples**

### **Example: Extract key information from an HTML string**

**Request**

```python
# pip install openai
 
from openai import OpenAI  # openai==1.52.2
 
client = OpenAI(
    api_key="up_FgMDhVt8dbVSOCV6GyBGix5ramkZE",
    base_url="https://api.upstage.ai/v1"
)
 
messages=[
    {
        "role": "system",
        "content": "You are an expert in information extraction. Extract information from the given HTML representation of image and organize them into a clear and accurate JSON format."
    },
    {
        "role": "user",
        "content": "HTML string: <table id='0' style='font-size:14px'><tr><td>1</td><td>FUTAMI 17 GREEN TEA (CLAS</td><td>12,500</td></tr><tr><td>1</td><td>EGG TART</td><td>13,000</td></tr><tr><td>1</td><td>GRAIN CROQUE MONSIEUR</td><td>17,000</td></tr></table><br><table id='1' style='font-size:18px'><tr><td>TOTAL</td><td>42, 500</td></tr><tr><td>CASH</td><td>50,000</td></tr><tr><td></td><td></td></tr><tr><td>CHANGE</td><td>7 ,500</td></tr></table>\n. Extract the structured data from the HTML string in JSON format."
    }
]
 
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "restaurant_receipt",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "menu_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "menu_cnt": {
                                "type": "number",
                                "description": "The count of the menu item."
                            },
                            "menu_name": {
                                "type": "string",
                                "description": "The name of the menu item."
                            },
                            "menu_price": {
                                "type": "number",
                                "description": "The price of the menu item."
                            }
                        },
                        "required": ["menu_cnt", "menu_name", "menu_price"],
                    }
                },
                "total_price": {
                    "type": "number",
                    "description": "The total price of the receipt."
                }
            },
            "required": ["menu_items", "total_price"],
        }
    }
}
 
response = client.chat.completions.create(
    model="solar-pro",
    messages=messages,
    response_format=response_format
)
 
print(response.choices[0].message.content)
```

**Response**

```python
{
    "menu_items": [
        {
           "menu_name": "FUTAMI 17 GREEN TEA (CLAS",
           "menu_price": 12500,
               "menu_cnt": 1
        },
        {
           "menu_name": "EGG TART",
           "menu_price": 13000,
               "menu_cnt": 1
        },
        {
           "menu_name": "GRAIN CROQUE MONSIEUR",
           "menu_price": 17000,
             "menu_cnt": 1
        }
    ],
    "total_price": 42500
}
```

# Embeddings

## [**What are embeddings?**](https://console.upstage.ai/docs/capabilities/embeddings#what-are-embeddings)

Embeddings are a way to convert text into vector representations, where semantically similar texts are positioned closely together. Embedding models can be used in various scenarios where a comparison of meaning between texts are needed, including search, classification, and clustering.

## [**What models does Upstage provide?**](https://console.upstage.ai/docs/capabilities/embeddings#what-models-does-upstage-provide)

The table below lists models currently available as an API. Upstage provides stable *aliases* that point to specific model versions, allowing you to integrate once and automatically benefit from future updates. We recommend using aliases instead of hardcoding model names, as models can be frequently updated.

| **Alias** | **Currently points to** | **RPM / TPM [(Learn more)](https://console.upstage.ai/docs/guides/rate-limits)** |
| --- | --- | --- |
| embedding-query | [solar-embedding-1-large-query](https://console.upstage.ai/docs/models#solar-embedding-1-large-query-beta) | 300 / 300,000 |
| embedding-passage | [solar-embedding-1-large-passage](https://console.upstage.ai/docs/models#solar-embedding-1-large-passage-beta) | 300 / 300,000 |

## [**What is the difference between passage and query models?**](https://console.upstage.ai/docs/capabilities/embeddings#what-is-the-difference-between-passage-and-query-models)

Both are converted into vectors to calculate similarity, but they serve different roles:

- **Passage**: Typically represents a portion of a document, a sentence, or longer text. It refers to a specific section or fragment of the text, often containing information. For example, a paragraph from a research paper or a description on a webpage can be used as a passage.
- **Query**: This is a question or search term that the user inputs to find information. It's usually shorter and more specific compared to a passage, conveying the user's intent to retrieve certain information.

## [**How to measure similarity between two texts**](https://console.upstage.ai/docs/capabilities/embeddings#how-to-measure-similarity-between-two-texts)

After converting text into embeddings, you can measure similarity between texts. Cosine similarity quantifies the similarity between two embedding vectors, representing the similarity between their corresponding texts. In the example below, dot products are used to calculate similarity. Note that since Upstage's Embeddings API outputs normalized vectors with a magnitude of 1, the dot product and cosine similarity yield the same result, and the passage corresponding to the vector with the highest similarity score is considered the most similar to the query.

```python
import numpy as np
 
similarity_list = []
for passage_embedding in passage_embedding_list:
  similarity = np.dot(passage_embedding, query_embedding)
  similarity_list.append(similarity)
 
most_similar_result = passage_list[np.argmax(similarity_list)]
print(most_similar_result)
```

## [**Examples**](https://console.upstage.ai/docs/capabilities/embeddings#examples)

### [**Example 1: Single text input**](https://console.upstage.ai/docs/capabilities/embeddings#example-1-single-text-input)

**Request**

```python
# pip install openai
 
from openai import OpenAI # openai==1.52.2
 
client = OpenAI(
    api_key="up_FgMDhVt8dbVSOCV6GyBGix5ramkZE",
    base_url="https://api.upstage.ai/v1"
)
 
response = client.embeddings.create(
    input="Solar embeddings are awesome",
    model="embedding-query"
)
 
print(response.data[0].embedding)
```

**Response**

```python
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        0.01850688,
        -0.0066606696,
        ...
        0.009938696,
        0.006452979
      ]
    }
  ],
  "model": "embedding-query",
  "usage": {
    "prompt_tokens": 21,
    "total_tokens": 21
  }
}
```

### [**Example 2: Batch processing**](https://console.upstage.ai/docs/capabilities/embeddings#example-2-batch-processing)

Upstage Embedding models support batch processing, allowing you to send an array of texts instead of a single input. This approach is typically faster and more efficient than processing texts one by one.

A single batch request can include up to 100 texts, with a total token limit of 204,800 per request.

```python
passage_list = [
    "Korea is a beautiful country to visit in the spring.",
    "The best time to visit Korea is in the fall.",
    "Best way to find bug is using unit test.",
    "Python is a great programming language for beginners.",
]
 
batch_process_result = client.embeddings.create(
    model = "embedding-passage",
    input = passage_list
).data
 
passage_embedding_list = [i.embedding for i in batch_process_result]
```

# Function calling

## [**What is function calling?**](https://console.upstage.ai/docs/capabilities/function-calling#what-is-function-calling)

Function calling enables your system to seamlessly interact with external services such as APIs, databases, or custom functions, transforming static models into dynamic, real-world tools. Developers can define custom functions within the tools array, specifying their purpose, inputs, and outputs. The model then dynamically generates function signatures in JSON format, unlocking a wide range of capabilities, including:

- **API calls**: LLMs can call APIs to retrieve real-time data, such as weather updates, stock prices, news, etc.
- **Database queries**: The model can interact with databases to pull specific data.
- **Automation tasks**: LLMs can trigger workflows or automation tools (like Zapier or other RPA platforms).
- **Code execution**: The model can generate code and trigger it using a function call for immediate execution.

## [**What models does Upstage provide?**](https://console.upstage.ai/docs/capabilities/function-calling#what-models-does-upstage-provide)

The table below lists models currently available as an API. Upstage provides stable *aliases* that point to specific model versions, allowing you to integrate once and automatically benefit from future updates. We recommend using aliases instead of hardcoding model names, as models can be frequently updated.

| **Alias** | **Currently points to** | **RPM / TPM [(Learn more)](https://console.upstage.ai/docs/guides/rate-limits)** |
| --- | --- | --- |
| solar-pro | [solar-pro-250422](https://console.upstage.ai/docs/models#solar-pro-250422) | 100 / 100,000 |
| solar-mini | [solar-mini-250422](https://console.upstage.ai/docs/models#solar-mini-250422) | 100 / 100,000 |

## [**Examples**](https://console.upstage.ai/docs/capabilities/function-calling#examples)

### [**Example 1: Request for weather data**](https://console.upstage.ai/docs/capabilities/function-calling#example-1-request-for-weather-data)

**Request**

```python
from openai import OpenAI  # openai==1.52.2
import json
 
client = OpenAI(
    api_key="up_FgMDhVt8dbVSOCV6GyBGix5ramkZE",
    base_url="https://api.upstage.ai/v1"
)
 
# Step 1: Setup and define the function
# This is an example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if unit is None:
        unit = "fahrenheit"
 
    if "seoul" in location.lower():
        return json.dumps({"location": "Seoul", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
 
# Step 2: Send the query and available functions to the model
def run_conversation():
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Seoul, and Paris?",
        }
    ]
 
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
 
    # Step 3: Check if the model has requested a function call
    # The model identifies that the query requires external data (e.g., real-time weather) and decides to call a relevant function, such as a weather API.
    response = client.chat.completions.create(
        model="solar-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
 
    # Step 4: Execute the function call
    # The JSON response from the model may not always be valid, so handle errors appropriately
    if tool_calls:
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # You can define multiple functions here as needed
        messages.append(response_message)  # Add the assistant's reply to the conversation history
 
        # Step 5: Process each function call and provide the results to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )  # Call the function with the provided arguments
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # Append the function response to the conversation history
 
        # Step 6: Generate a new response from the model using the updated conversation history
        second_response = client.chat.completions.create(
            model="solar-mini",
            messages=messages,
        )
        return second_response  # Return the final response from the model
 
response = run_conversation()
print(response.choices[0].message.content)
```

**Response**

```python
The current weather in San Francisco is 72 degrees Celsius, in Seoul is 10 degrees Celsius, and in Paris is 22 degrees Celsius.
```

### [**Example 2: Retrieving information from structured data**](https://console.upstage.ai/docs/capabilities/function-calling#example-2-retrieving-information-from-structured-data)

**Request**

```python
import json
import os
from typing import List
 
from openai import OpenAI  # openai==1.52.2
import pandas as pd
 
 
client = OpenAI(
    api_key="up_FgMDhVt8dbVSOCV6GyBGix5ramkZE",
    base_url="https://api.upstage.ai/v1"
)
 
 
# Step 1. Setup and create structured data
# This is a hard coded dummy data to demonstrate the function call capability
# In production, this could be your internal database
 
# Create data(MAU and revenue)
mau_data = [
    {"Month": 1, "Service": "LLM", "MAU": 100},
    {"Month": 1, "Service": "Embedding", "MAU": 50},
    {"Month": 2, "Service": "LLM", "MAU": 150},
    {"Month": 2, "Service": "Embedding", "MAU": 70},
    {"Month": 3, "Service": "LLM", "MAU": 300},
    {"Month": 3, "Service": "Embedding", "MAU": 80},
    {"Month": 4, "Service": "LLM", "MAU": 350},
    {"Month": 4, "Service": "Embedding", "MAU": 150},
]
mau = pd.DataFrame(mau_data)
 
revenue_data = [
    {"Month": 1, "Service": "LLM", "Revenue": 1000},
    {"Month": 1, "Service": "Embedding", "Revenue": 500},
    {"Month": 2, "Service": "LLM", "Revenue": 1500},
    {"Month": 2, "Service": "Embedding", "Revenue": 700},
    {"Month": 3, "Service": "LLM", "Revenue": 3000},
    {"Month": 3, "Service": "Embedding", "Revenue": 800},
    {"Month": 4, "Service": "LLM", "Revenue": 3500},
    {"Month": 4, "Service": "Embedding", "Revenue": 1500},
]
revenue = pd.DataFrame(revenue_data)
 
 
# Step 2. Define the function
# This function query_data is designed to filter data from either the MAU or Revenue tables based on the months and service specified.
# It returns a filtered DataFrame based on these criteria.
# This function will be later utilized by the LLM as part of the function call mechanism to dynamically answer queries.
 
# Example question
question = "How much revenue occured in 1st quarter for LLM?"
 
def query_data(table_name: str, month: List[int], service: str) -> pd.DataFrame:
    """
    Query data from the given table based on month and service.
    """
    if table_name == "MAU":
        data = mau
    elif table_name == "Revenue":
        data = revenue
    else:
        raise ValueError(f"Table name {table_name} not found.")
 
    return data[(data["Month"].isin(month)) & (data["Service"] == service)]
 
 
# Step 3. Define tools for function calling
# This block defines a tools array that registers the query_data function as an available option for the LLM.
# It provides the model with information about the available functions and its expected parameters, enabling the model to dynamically call the functions during conversations.
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": "query data from the table based on month and service",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The table name to query.",
                        "enum": ["MAU", "Revenue"],
                    },
                    "month": {
                        "type": "array",
                        "items": {"type": "int"},
                        "description": "The list of months to query.",
                    },
                    "service": {
                        "type": "string",
                        "description": "Which service to query.",
                        "enum": ["LLM", "Embedding"],
                    },
                },
            },
        },
    }
]
 
 
# Step 4. User message and initial LLM response
# The LLM receives a user message inquiring about the revenue in the 1st quarter for LLM. Along with this, the tools list (including query_data) provided, enabling the model to decide whether a function call is needed.
# If the model determines that the query_data function should be used, it will generate a request, which will be handled in the next step.
 
messages = [
    {
        "role": "user",
        "content": question,
    }
]
 
response = client.chat.completions.create(
    model="solar-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
 
 
# Step 5. Function execution and tool call handling
# The LLM’s response includes a tool call (tool_call), indicating the model has chosen to call the query_data function.
# The tool call contains the function name and arguments, which are extracted and parsed.
# The query_data function is executed using the extracted arguments.
 
# Extract the tool call from the LLM's response
tool_call = response.choices[0].message.tool_calls[0]
 
# Define the available functions
available_functions = {
    "query_data": query_data,
}
 
# Retrieve the function name and corresponding callable function
function_name = tool_call.function.name
function_to_call = available_functions[function_name]
 
# Parse the function arguments from the tool call
function_args = json.loads(tool_call.function.arguments)
 
# Execute the function with the parsed arguments
function_response = function_to_call(**function_args)
function_response = json.dumps(function_response.to_dict(orient="records"))
 
 
# Step 6. Extend the conversation with function output
# The function’s output is appended to the conversation as a message.
# The model is then prompted to continue the conversation, now informed by the function's results.
 
# Append the function's response to the conversation history
messages.append(
    {
        "role": "tool",
        "name": function_name,
        "content": function_response,
    }
)
 
# Continue the conversation by sending the updated message history to the LLM
second_response = client.chat.completions.create(
    model="solar-mini",
    messages=messages,
)
 
print(second_response.choices[0].message.content)
```

**Response**

```python
The revenue for LLM in the 1st quarter was $1000 in January, $1500 in February, and $3000 in March.
```

# Groundedness checking

## [**What is groundedness checking?**](https://console.upstage.ai/docs/capabilities/groundedness-checking#what-is-groundedness-checking)

Large Language Models (LLMs) are capable of generating detailed, information-rich content, but they are also prone to hallucinations—producing factually incorrect or ungrounded responses. One popular approach to address this limitation is to provide LLMs with reference text, often called "context," which helps generate more accurate outputs. This method is known as Retrieval Augmented Generation (RAG).

While RAG enhances the accuracy of LLM outputs by grounding them in provided context, it doesn't always guarantee truthfulness. Therefore, an additional verification step is necessary to ensure the model’s response aligns with the given context. The Groundedness check API is designed precisely for this purpose: it evaluates whether a model-generated response is grounded in the provided context. Based on this check, the API will classify the response as grounded, not grounded, or uncertain.

Incorporating the Groundedness Check API significantly improves the reliability of the system. By validating responses before they are delivered, it helps ensure factual accuracy, reducing the risk of hallucinations to nearly zero.

## [**What models does Upstage provide?**](https://console.upstage.ai/docs/capabilities/groundedness-checking#what-models-does-upstage-provide)

The table below lists models currently available as an API. Upstage provides stable *aliases* that point to specific model versions, allowing you to integrate once and automatically benefit from future updates. We recommend using aliases instead of hardcoding model names, as models can be frequently updated.

| **Alias** | **Currently points to** | **RPM / TPM [(Learn more)](https://console.upstage.ai/docs/guides/rate-limits)** |
| --- | --- | --- |
| groundedness-check | [groundedness-check-240502](https://console.upstage.ai/docs/models#groundedness-check-240502-beta) | 100 / 100,000 |

### [**With and without groundedness check**](https://console.upstage.ai/docs/capabilities/groundedness-checking#with-and-without-groundedness-check)

While RAG significantly addresses the hallucination issue in LLMs, Groundedness check acts as a safeguard to confirm the accuracy of the response. Reflecting on the example provided, RAG alone might not prevent an initial inaccurate response from being presented to the user as shown in the first case. However, Groundedness check allows for the response to be redirected to the LLM, offering a chance for correction.

This process of Groundedness check meticulously checks if the output **aligns with the content from the reference document**. By ensuring the answer is consistent with the retrieved data, it verifies that the model's output is firmly anchored in the provided context, thereby eliminating scenarios where the response is unmoored from the given information.

### [**Parameters**](https://console.upstage.ai/docs/capabilities/groundedness-checking#parameters)

- The `messages` parameter should be a list of message objects containing two elements:
    - a user-provided context and
    - an assistant's response to be checked.
- Each message object must specify the type of message as either `user` (context) or `assistant` (response) using the `role` attribute, and set the `content` attribute with the corresponding text string.

The API response will be a string with a value of either `grounded`, `notGrounded`, or `notSure`. The `notSure` response is returned when the groundedness of the assistant's response to the provided context cannot be clearly determined.

## [**Example**](https://console.upstage.ai/docs/capabilities/groundedness-checking#example)

### [**Request**](https://console.upstage.ai/docs/capabilities/groundedness-checking#request)

```python
# pip install openai
 
from openai import OpenAI # openai == 1.2.0
 
client = OpenAI(
    api_key="up_FgMDhVt8dbVSOCV6GyBGix5ramkZE",
    base_url="https://api.upstage.ai/v1"
)
 
response = client.chat.completions.create(
    model="groundedness-check",
    messages=[
        {
            "role": "user",
            "content": "Mauna Kea is an inactive volcano on the island of Hawaiʻi. Its peak is 4,207.3 m above sea level, making it the highest point in Hawaii and second-highest peak of an island on Earth."
        },
        {
            "role": "assistant",
            "content": "Mauna Kea is 5,207.3 meters tall."
        }
    ]
)
 
print(response)
```

### [**Response**](https://console.upstage.ai/docs/capabilities/groundedness-checking#response)

```python
{
  "id": "c43ecfa6-31a9-4884-a920-a5f44fb727df",
  "object": "chat.completion",
  "created": 1710338020,
  "model": "groundedness-check-240502",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "notGrounded"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 132,
    "completion_tokens": 3,
    "total_tokens": 135
  },
  "system_fingerprint": ""
}# pip install openai
 
from openai import OpenAI # openai == 1.2.0
 
client = OpenAI(
    api_key="up_FgMDhVt8dbVSOCV6GyBGix5ramkZE",
    base_url="https://api.upstage.ai/v1"
)
 
response = client.chat.completions.create(
    model="groundedness-check",
    messages=[
        {
            "role": "user",
            "content": "Mauna Kea is an inactive volcano on the island of Hawaiʻi. Its peak is 4,207.3 m above sea level, making it the highest point in Hawaii and second-highest peak of an island on Earth."
        },
        {
            "role": "assistant",
            "content": "Mauna Kea is 5,207.3 meters tall."
        }
    ]
)
 
print(response)
```