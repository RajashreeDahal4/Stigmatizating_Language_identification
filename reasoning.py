
import os
import json
from player import Members
import re
import ast
import os
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
HF_token=os.getenv("HF_TOKEN")
model_path="/home/rdaha1@AD/bias/Multi-Agents-Debate/models/Llama-3.3-70B-Instruct"


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained(model_path,use_auth_token=HF_token)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # Use float32 if you don't have enough VRAM
    device_map="auto",
    use_auth_token=HF_token  # Automatically assigns to GPU if available
)


NAME_LIST=[
    "Affirmative side",
    "Negative side",
    "Moderator",
]

class DebatePlayer(Members):
    def __init__(self, model_name: str, name: str, temperature:float, sleep_time: float) -> None:
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            sleep_time (float): sleep because of rate limits
        """
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time)


class Debate:
    def __init__(self,
            model_name: str="/home/rdaha1@AD/bias/Multi-Agents-Debate/models/Llama-3.3-70B-Instruct", 
            temperature: float=0.2, 
            num_players: int=3, 
            config: dict=None,
            max_round: int=2,
            sleep_time: float=0
        ) -> None:
        """Create a debate

        Args:
            model_name (str):  model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """

        self.model_name="/home/rdaha1@AD/bias/Multi-Agents-Debate/models/Llama-3.3-70B-Instruct"
        self.temperature = temperature
        self.num_players = num_players
        self.config = config
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.tracking_answers={}

        self.init_prompt()
        self.creat_agents()
        self.init_agents()


    def init_prompt(self):
        def prompt_replace(key):
            self.config[key] = self.config[key].replace("##debate_topic##", self.config["debate_topic"])
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")

    def creat_agents(self):
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature, sleep_time=self.sleep_time) for name in NAME_LIST
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        self.affirmative.set_meta_prompt(self.config['player_meta_prompt'])
        self.negative.set_meta_prompt(self.config['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])
        self.affirmative.add_event(self.config['affirmative_prompt'])
        self.aff_ans = self.affirmative.ask(tokenizer,model)
        self.affirmative.add_memory(self.aff_ans)
        self.config['base_answer'] = self.aff_ans

        self.negative.add_event(self.config['negative_prompt'].replace('##aff_ans##', self.aff_ans))
        self.neg_ans = self.negative.ask(tokenizer,model)

        self.negative.add_memory(self.neg_ans)

        self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', 'first'))
        self.mod_ans = self.moderator.ask(tokenizer,model)
        match = re.search(r'\{.*?\}', self.mod_ans, re.DOTALL)
        self.mod_ans = json.loads(match.group()) 

        self.tracking_answers["round1"]=self.mod_ans

        self.moderator.add_memory(self.mod_ans)

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def print_answer(self):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config["debate_topic"])
        print("\n----- Base Answer -----")
        print(self.config["base_answer"])
        print("\n----- Debate Answer -----")
        print(self.config["debate_answer"])
        print("\n----- Debate Reason -----")
        print(self.config["Reason"])

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players. 

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: DebatePlayer):
        ans = player.ask(tokenizer,model)
        player.add_memory(ans)
        self.speak(player.name, ans)


    def run(self):
        self.mod_ans['debate_answer']=""
        for round in range(self.max_round - 1):
            if self.mod_ans["debate_answer"]!="":
                break
            else:
                print(f"===== Debate Round-{round+2} =====\n")
                self.affirmative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
                self.aff_ans = self.affirmative.ask(tokenizer,model)
                self.affirmative.add_memory(self.aff_ans)

                self.negative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
                self.neg_ans = self.negative.ask(tokenizer,model)
                self.negative.add_memory(self.neg_ans)

                self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', self.round_dct(round+2)))
                mod_ans = self.moderator.ask(tokenizer,model)
                mod_ans=mod_ans.replace("'s",'')
                match = re.search(r'\{.*?\}', mod_ans, re.DOTALL)
                if match:
                    mod_ans = ast.literal_eval(match.group())
                    print("mod_answer",mod_ans)
                else:
                    mod_ans=mod_ans 
                    print("==mod_answer",mod_ans)

                mod_ans = json.loads(json.dumps(mod_ans))
                self.mod_ans.update(mod_ans)
                self.tracking_answers[f"round{round+2}"]=self.mod_ans

                self.moderator.add_memory(self.mod_ans)
        if self.mod_ans["Consensus"]=="False":
            judge_ans={"debate_answer":[]}
            print("INSERTING JUDGE",self.mod_ans)
            judge_player = DebatePlayer(model_name=self.model_name, name='Judge', temperature=self.temperature, sleep_time=self.sleep_time)
            aff_ans = self.affirmative.memory_lst[2]['content']
            neg_ans = self.negative.memory_lst[2]['content']

            judge_player.set_meta_prompt(self.config['moderator_meta_prompt'])

            # extract answer candidates
            judge_player.add_event(self.config['judge_prompt_last1'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            ans = judge_player.ask(tokenizer,model)
            print("Judge1==========================================",ans)

            judge_player.add_memory(ans)

            # select one from the candidates
            judge_player.add_event(self.config['judge_prompt_last2'])
            ans = judge_player.ask(tokenizer,model)
            match = re.search(r'\{.*?\}', ans, re.DOTALL)
            if match:
                try:
                    ans = ast.literal_eval(match.group())
                    ans['debate_answer']=''
                except:
                    ans=ans
            print("Judge2==========================================",ans)
            judge_ans.update(ans)
            self.tracking_answers['Judge']=judge_ans
            judge_player.add_memory(ans)
            
            if ans["debate_answer"] != '':
                self.config['success'] = True
                # save file
            self.config.update(ans)
            self.players.append(judge_player)

        return self.tracking_answers
bias_types={"Explicit Bias": "Explicit forms of bias include preferences, beliefs, and attitudes of which people are generally consciously aware, endorsed, and can be identified and communicated (e.g., labeling a patient as 'non-compliant' or 'irresponsible').",
            "Implicit Bias": "Implicit biases are unconscious mental processes that lead to associations and reactions that are automatic and without intention and actors have no awareness of the associations with a stimulus.  (e.g., noting that a patient is 'well-spoken for their background').",
            "Stigmatizing Language": "Stigmatizing language is any language in clinical documentation that portrays the patient in a negative or judgmental way—by expressing doubt about their credibility, implying they are difficult, noncompliant, or deceptive, or by using language that casts their behavior, attitude, or emotional state in a negative light. This includes both overt and subtle wording that reflects blame, shame, or disrespect. Example: 'claims to see a therapist'",
            "Racial Bias": "Racism is an organized social system in which the dominant racial group, based on an ideology of inferiority, categorizes and ranks people into social groups called ‘races’ and uses its power to devalue, disempower, and differentially allocate valued society resources and opportunities to groups defined as inferior. Example: Black male, non-compliant with medications.",
            "Gender Bias": "Gender bias refers to both assuming sameness when differences exist and assuming differences when none exist, leading to unequal treatment or stereotyping. Example: 'patient is a middle-aged woman with a demanding family life"
}

if __name__ == "__main__":

    path="/home/rdaha1@AD/bias/Multi-Agents-Debate/dumps/llama70_mimic"
    files=os.listdir(path)
    full_files=[path+"/"+i for i in files]
    for enum, each_file in enumerate(full_files):
        dump_file = each_file.replace("llama70_mimic", "llama70_mimic_reasoning")

        # Load full debate data
        with open(each_file, 'rb') as file:
            debate_data = json.load(file)

        # Load existing dump if available
        if os.path.exists(dump_file):
            with open(dump_file, 'r') as f:
                reasoning_data = json.load(f)
        else:
            reasoning_data = []

        # Determine how many have already been processed
        processed_count = len(reasoning_data)

        if processed_count >= len(debate_data):
            print(f"Skipping {dump_file}: already processed.")
            continue

        # Resume processing from the next unprocessed item
        for i in range(processed_count, len(debate_data)):
            each_data = debate_data[i]

            debate_topic = f"""
            Paragraph: {each_data['chunk_paragraph']}.
            Identified bias is: {each_data['bias_type']}.
            Definition of {each_data['bias_type']}: {bias_types[each_data['bias_type']]}
            Quote: {each_data['quote']}.
            Debate Initiation: We will now begin a debate based on the identified bias, and supporting quote, and given paragraph.
            """

            config = json.load(open("/home/rdaha1@AD/bias/Multi-Agents-Debate/code/utils/config", "r"))
            config['debate_topic'] = debate_topic

            print("============================ DEBATE TOPIC =========================================")
            print(debate_topic)
            print("\n")

            debate = Debate(num_players=3, config=config, temperature=0.1, sleep_time=0)
            result = debate.run()

            each_data["reasoning"] = result
            reasoning_data.append(each_data)

            # Save after each entry to prevent loss in case of crash
            with open(dump_file, 'w') as outfile:
                json.dump(reasoning_data, outfile, indent=4)