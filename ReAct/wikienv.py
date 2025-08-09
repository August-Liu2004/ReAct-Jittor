import ast
import json
import time
import gym
import requests
from bs4 import BeautifulSoup
import re
from openai import OpenAI

def deepseek_translator(thought:str,text: str) -> str:
    client = OpenAI(
        api_key="sk-e6dca9c023dd455291b4946c5f89e171",
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
           {"role": "system", "content": "你是一名专业翻译，仅翻译 text。不要解释或输出任何额外内容，仅返回翻译结果。"},
           {"role": "user", "content": f"请翻译以下 text，\n{text}"}
        ],
        stream=False
    )

    return response.choices[0].message.content.strip()
def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace('\xa0', '').replace('\u3000', '').replace('&nbsp;', '')
    s = re.sub(r"[\[\【][0-9a-zA-Z一-龥]{1,10}[\]\】]", "", s)
    return s.strip()
def extract_sentences(text):
    paragraphs = text.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    sentences = []
    for p in paragraphs:
        sentences += p.split('。')  # 中文句号
    return [s.strip() + '。' for s in sentences if s.strip()]

class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class WikiEnv(gym.Env):

  def __init__(self):
    """
      Initialize the environment.
    """
    super().__init__()
    self.page = None  # current Wikipedia page
    self.obs = None  # current observation
    self.lookup_keyword = None  # current lookup keyword
    self.lookup_list = None  # list of paragraphs containing current lookup keyword
    self.lookup_cnt = None  # current lookup index
    self.steps = 0  # current number of steps
    self.answer = None  # current answer from the agent
    self.observation_space = self.action_space = textSpace()
    self.search_time = 0
    self.num_searches = 0
    self.sentences=None
    
  def _get_obs(self):
    return self.obs

  def _get_info(self):
    return {"steps": self.steps, "answer": self.answer}

  def reset(self, seed=None, return_info=False, options=None):
    # We need the following line to seed self.np_random
    # super().reset(seed=seed)
    self.obs = ("Interact with BaiduBaike using search[], lookup[], and "
                "finish[].\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    self.sentences=None
    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation

  def construct_lookup_list(self, keyword):
    # find all paragraphs
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  # @staticmethod
  # def get_page_obs(page):
  #   # find all paragraphs
  #   paragraphs = page.split("\n")
  #   paragraphs = [p.strip() for p in paragraphs if p.strip()]

  #   # find all sentence
  #   sentences = []
  #   for p in paragraphs:
  #     sentences += p.split('. ')
  #   sentences = [s.strip() + '.' for s in sentences if s.strip()]
  #   return ' '.join(sentences[:5])
  def search_step(self, entity):
        entity_ = entity.strip().replace(" ", "+")
        url = f"https://baike.baidu.com/item/{entity_}"

        print(f"🔍 正在搜索：{entity}...")
        start = time.time()
        try:
            response = requests.get(url)
            self.search_time += time.time() - start
            self.num_searches += 1

            if response.status_code == 404:
                self.page = None
                self.obs = f"❌ "
                return

            soup = BeautifulSoup(response.text, "html.parser")
            desc_tag = soup.find("div", class_="J-lemma-content")
            desc_intro = soup.find("div", class_=re.compile(r"lemmaSummary_(\w+)\sJ-summary"))

            if desc_tag and desc_intro:
                content = desc_intro.get_text().strip() + "\n" + desc_tag.get_text().strip()
                self.sentences = extract_sentences(content)
                self.sentences = [normalize_text(s) for s in self.sentences[:5]]
                self.page = self.sentences
                self.obs = "✅ 已加载百科内容：\n" + ''.join(self.sentences)
                self.lookup_keyword = None
                self.lookup_list = None
                self.lookup_cnt = None
            else:
                self.page = None
                self.obs = f"❗ 未找到 {entity} 的简介，或页面结构已变。"

        except Exception as e:
            self.page = None
            self.obs = f"🚫 百度百科请求异常：{e}"
        
  
  def step(self, action):
        reward = 0
        done = False
        action = action.strip()
        thought = getattr(self, "thought", None)
        print("这次拿到的action:",repr(action))

        if self.answer is not None:  # already finished
            done = True
            return self.obs, reward, done, self._get_info()

        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            entity=deepseek_translator(thought,entity)
            print("这次拿到的entity是",entity)
            self.search_step(entity)

        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1].strip()
            keyword=deepseek_translator(thought,keyword)
            print("这次拿到的keyword是",keyword)
            keyword = normalize_text(keyword)

            if not self.page:
                self.obs = "please search[...] first。"
            else:
                if self.lookup_keyword != keyword:  # reset
                    self.lookup_keyword = keyword
                    self.lookup_list = [
                        s for s in self.sentences
                        
                        if keyword in normalize_text(s)
                    ]
                    self.lookup_cnt = 0

                if not self.lookup_list:
                    self.obs = f"🔍 cannot find “{keyword}” 。"
                elif self.lookup_cnt >= len(self.lookup_list):
                    self.obs = "🔚 no more sentences。\n"
                else:
                    result = self.lookup_list[self.lookup_cnt]
                    self.lookup_cnt += 1
                    self.obs = f"(Result {self.lookup_cnt} / {len(self.lookup_list)}) {result}"

        elif action.startswith("finish[") :
          match = re.search(r'finish\[(.*?)\]', action, re.IGNORECASE)
          if match:
            answer =match.group(1).strip()
            print("已经结束咧！",answer)
            if self.answer is None:
             self.answer = answer
            done = True
            self.obs = answer
            


        self.steps += 1
        return self.obs, reward, done, self._get_info()
  
  def get_time_info(self):
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }
