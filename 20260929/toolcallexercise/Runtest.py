import os
from openai import OpenAI

settings={
    "reasoning_effort":"high",
    "base_url":"https://api.deepseek.com",
    "api_key":"",
    "model":"deepseek-flash",
    "stream":False,
    "sys_prompt":"当前用户是一名学生，正在学习调用模型接口。仅根据用户要求进行回应，不需要自行扩展。"
}

message_list=[]
user_input=""

def reset_settings():
    settings["reasoning_effort"]="low" #low,high,max
    settings["base_url"]="https://api.deepseek.com"
    settings["api_key"]=""
    settings["model"]="deepseek-flash"
    settings["stream"]=False
    return 0

def new():
    message_list=[]
    message_list.append({"role": "system", "content": f"{settings["sys_prompt"]}"})
    return 0

def reset_system_prompt():
    settings["sys_ptompt"]="当前用户是一名学生，正在学习调用模型接口。仅根据用户要求进行回应，不需要自行扩展。"
    return 0

def text_in():
    user_input=input("Please text in:")
    if user_input=="end":
        return "end"
    message_list.append({"role":"user","content":f"{user_input}"})
    return 0

def request_up():
    response = client.chat.completions.create(
        model=settings["model"],
        messages=message_list,
        stream=settings["stream"],
        reasoning_effort=settings["reasoning_effort"],
        extra_body={"thinking": {"type": "enabled"}}
    )
    print(response.choices[0].message.content)
    message_list.append(response.choices[0].message)


try:
    reset_settings()
    reset_system_prompt()
    client = OpenAI(
        api_key=settings["api_key"],
        base_url=settings["base_url"])
    new()
    while(1):
        if(text_in()=="end"):
            break
        request_up()
except Exception as e:
    print(f"Error Info:{e}")

print("Run ends.")