import os, types
import json
from enum import Enum
import requests
import time
import zhipuai
import litellm
from typing import Callable, Optional
from litellm.utils import ModelResponse

class ZhipuError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class ChatClmConfig():
    max_length: Optional[int]=None
    max_new_tokens: Optional[int]=litellm.max_tokens # petals requires max tokens to be set
    do_sample: Optional[bool]=None
    temperature: Optional[float]=None
    top_k: Optional[int]=None
    top_p: Optional[float]=None
    repetition_penalty: Optional[float]=None

    def __init__(self,
                 max_length: Optional[int]=None,
                 max_new_tokens: Optional[int]=litellm.max_tokens, # petals requires max tokens to be set
                 do_sample: Optional[bool]=None,
                 temperature: Optional[float]=None,
                 top_k: Optional[int]=None,
                 top_p: Optional[float]=None,
                 repetition_penalty: Optional[float]=None) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
    
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}

def validate_environment(api_key):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    # if api_key:
    #     headers["Authorization"] = f"Api-Key {api_key}"
    return headers

def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params=None,
    stream=False,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    model = model
    prompt = ""
    for message in messages:
        if "role" in message:
            if message["role"] == "user":
                prompt += f"{message['content']}"
            else:
                prompt += f"{message['content']}"
        else:
            prompt += f"{message['content']}"

    ## Load Config
    config = litellm.ChatClmConfig.get_config() 
    for k, v in config.items(): 
        if k not in optional_params: # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    data = {
        "model": model,
        "prompt": prompt,
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
        )
    ## COMPLETION CALL
	#zhipuai.api_key = api_key
    zhipuai.api_key = "38dc046c622b6ec1b9bfa0413e6ca2ee.2Yn3uFr8j74pMOvK"
    
    if stream:
        response = zhipuai.model_api.async_invoke(**data)
    else:
        response = zhipuai.model_api.invoke(**data)
    
    if "stream" in optional_params and optional_params["stream"] == True:
        return response.iter_lines()
    else:
        ## LOGGING
        logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                original_response=response,
                additional_args={"complete_input_dict": data},
            )
        completion_response = response
        if "error" in completion_response:
            raise ZhipuError(
                message=completion_response["error"],
                status_code=response.status_code,
            )
        else:
            print_verbose(f"raw model_response: {completion_response}")
            if "data" not in completion_response:
                raise ZhipuError(
                    message=f"Unable to parse response. Original response: {response.text}",
                    status_code=response.status_code
                )
        model_response["created"] = time.time()
        model_response["model"] = model
        model_response["usage"] = completion_response["data"]['usage']
        res_content = completion_response["data"]['choices'][0]['content']
        res_content = res_content.strip(' "')
        res_content = json.loads(f'"{res_content}"')
        model_response["choices"][0]["message"]["content"] = res_content
        return model_response

def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
