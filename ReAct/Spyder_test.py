import time
import requests
from bs4 import BeautifulSoup
import re
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

class BaiduBaikeAgent:
    def __init__(self):
        super().__init__()
        self.page = None                  # 当前页面内容（原始 text）
        self.obs = None                   # 当前 observation 输出
        self.lookup_keyword = None       # 当前 lookup 关键词
        self.lookup_list = None          # 命中句子列表
        self.lookup_cnt = None           # 当前 lookup 索引
        self.steps = 0                   # 步数
        self.answer = None               # finish[] 的最终回答
        self.search_time = 0             # 累计搜索时间
        self.num_searches = 0 # 搜索次数
        self.sentences=None           
    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}
    def search_step(self, entity):
        entity_ = entity.strip().replace(" ", "+")
        url = f"https://baike.baidu.com/item/{entity_}"

        print(f"🔍 正在搜索：{entity}...")
        start = time.time()
        try:
            response = requests.get(url)
            print(response)
            self.search_time += time.time() - start
            self.num_searches += 1

            if response.status_code != 200:
                self.page = None
                self.obs = f"❌ 请求失败，状态码：{response.status_code}"
                return

            soup = BeautifulSoup(response.text, "html.parser")
            desc_tag = soup.find("div", class_="J-lemma-content")
            desc_intro = soup.find("div", class_="lemmaSummary_Ydljy J-summary")

            if desc_tag and desc_intro:
                content = desc_intro.get_text().strip() + "\n" + desc_tag.get_text().strip()
                self.sentences = extract_sentences(content)
                self.sentences = [normalize_text(s) for s in self.sentences]
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

        if self.answer is not None:  # already finished
            done = True
            return self.obs, reward, done, self._get_info()

        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            self.search_step(entity)

        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1].strip()
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

        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            self.obs = f"🏁 task over,the answer is：{answer}"

        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "🧠 Thought acknowledged."

        else:
            self.obs = f"🚫illegal action：{action}"

        self.steps += 1
        return self.obs, reward, done, self._get_info()

# ✅ 示例交互
if __name__ == "__main__":
    agent = BaiduBaikeAgent()
    print("🧪 百度百科 Agent 启动。支持指令：search[...]、lookup[...]、finish[...]\n")

    while True:
        action = input(">>> ")
        if action.lower() in ["exit", "quit"]:
            print("👋 已退出。")
            break
        output,reward,done,info = agent.step(action)
        print(output,reward,done,info)
        if done:
            break
