# GigaMemory: global memory for LLM
## Contents
- [Task description](#task-description)
- [Description of the solution format](#description-of-the-solution-format)
- [Data](#data)
- [Metric calculation](#metric-calculation)
- [Restrictions](#restrictions)
- [Baseline solution](#baseline-solution)
- [The prize pool](#the-prize-pool)
## Task description
Large Language Model (LLM) is a hot topic for research and development in recent years. However, LLMs, despite all their power, still face a major problem — they have poorly developed memory. When communicating with a user, the model forgets even basic information about this person the next day: his name, age, etc. This significantly worsens the user experience and limits the possibilities of using LLM in various applications. However, over the past year, the first successes in the development of this direction have appeared, which gives a powerful impact to LLM memory research. How to memorize data? How and in what format should the memorized information be stored? How to use the selected data during LLM response generation? As part of this competition, participants are invited to answer these and many other applied questions about the memory module development.


### Statement of the task

The essence of the task is to create global memory for LLM in the form of an independent module. Global memory refers to the ability to extract and remember atomic facts about a user, containing any information that may be useful in further interactions with the user.
Any technology within the competition restrictions may be used to implement the memory module.

The format and type of data in which the memory is stored are arbitrary.

The memory module's performance quality will be tested on the dialogue dataset: the validation and test parts. Until the solution making process is complete, only the validation part will be used, on the basis of which the public leaderboard is considered; after that will be used the test part, which in turn forms the private leaderboard. Both parts of the dataset are closed.

The principle of checking the memory module for each dialogue is as follows: the dialogue will be divided into pairs of user replica and assistant response. The method of writing to memory is successively called on new dialogue pairs and must store information about the user. Thus, n/2 memory calls are made for each dialogue, where n is the length of the dialogue. The memory module generates memory of any format and data type. 
After going through the entire dialogue, the assistant's method of answering the test question is called using memory. The participant independently forms a prompt for the model [GigaChat Lite](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16), which must answer the question. The prompt is expected to contain the memory object, the question, and all necessary instructions for the answer.
The received answer is submitted for evaluation of compliance with the correct answer using "LLM as a judge".

## Description of the solution format

The code packed in a ZIP archive must be sent to the verification system. 
Solutions are run in an isolated environment using Docker. Time and resources are limited during testing (see the ["Restrictions" section](#restrictions)).

An example of the solution being sent (see the ["Baseline solution" section](#baseline-solution)) can be found in the repository in the [src/submit directory](https://github.com/ai-forever/memory_aij2025/tree/main/src/submit). To create a test submit, you need to pack the contents of this directory into a ZIP archive and send it to the testing system.

You can reproduce the existing environment to run solutions locally. The Dockerfile and requirements.txt for this image are provided in the repository.

Important points for creating the solution:
* All necessary libraries that are not in the base image are suggested to be downloaded in the solution archive and installed from the code (taking into account compatibility with the libraries inside the base image)
* Weights of small-sized models that satisfy the total solution volume limit of 5 GB can be downloaded in the solution itself
* The version of the base model [GigaChat Lite](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16) on which the solution is supposed to be built is already placed in the base image at the /app/models path.

The structure of the code of the solution itself is not strictly regulated. It is only necessary that the structure of the solution contains:
* the model_inference.py script with the implementation of the SubmitModelWithMemory class, which will correspond to the signature of the abstract [ModelWithMemory](https://github.com/ai-forever/memory_aij2025/blob/main/src/submit_interface.py) class with the implementation of all the necessary methods from it:
```python
class ModelWithMemory(ABC):

   @abstractmethod
   def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
       # write to memory
       pass

   @abstractmethod
   def clear_memory(self, dialogue_id: str) -> None:
       # clear memory
       pass

   @abstractmethod
   def answer_to_question(self, rq_id: str, question: str) -> str:
       # get answer to question using GigaChat Lite
       pass
```
- script \_\_init\_\_.py with import of this class
It is allowed to add other files and code necessary for the implementation of the solution. It is not allowed to change other classes and methods of the organizers' code.
An example of the correct implementation of a class and methods can be found in the Baseline solution.

To verify the correctness of your solution locally, it is suggested to perform the following steps:
1. Clone the current repository and go to the root directory
2. Clone the base model to yourself:
```
git lfs install
git clone https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16
```
2.1. Make sure the model is loaded into the GigaChat-20B-A3B-instruct-v1.5-bf16 directory (the files take up ~40 GB in total)
```
cd GigaChat-20B-A3B-instruct-v1.5-bf16
ls -lha
cd ..
```
2.2. In case of incorrect loading:
```
cd GigaChat-20B-A3B-instruct-v1.5-bf16
git lfs pull
cd ..
```
3. Create an environment:
```
conda create -n my_env python=3.10 -y
```
4. Activate the environment:
```
conda activate my_env
```
5. Install the required minimum dependencies:
```
pip install -r requirements.txt
```
6. The entry point for launching the baseline is the file [src/run.py](https://github.com/ai-forever/memory_aij2025/blob/main/src/run.py). To launch correctly, you need to specify the following paths in the start method (lines 165-170):
    - dataset_path - the path to the dataset from which the model's predictions will be calculated. [A sample dataset is posted in the repository](https://github.com/ai-forever/memory_aij2025/blob/main/data/format_example.jsonl). To run with it, specify the path `../data/format_example.jsonl`
    - output_path is the path to the directory where the results of the model will be saved. If there is no directory, it will be created automatically.
    - model_path is the path to the directory where the GigaChat Lite model is located.

Code example with relative paths:
```python
if __name__ == "__main__":
    launch_inference_and_check_errors(
        dataset_path="../data/format_example.jsonl",
        output_path="../output",
        model_path="../GigaChat-20B-A3B-instruct-v1.5-bf16"
    )
```
7. Run the inference baseline from the root directory of the repository:
```
cd src && python run.py
```

## Data
### Input Data

A dataset was created for the Competition to evaluate the ability of language models to retain long-term memory in the context of multi-session interactions. It includes dialogues between a user and an assistant.

The objective is to assess how effectively the memory module can store and retrieve information from lengthy conversations. To facilitate this, the dataset includes questions about the dialogue history. On average, each dialogue consists of several dozen sessions of varying lengths and contains roughly 300,000 characters (or about 100,000 tokens).

The complete evaluation dataset is not available to Participants. Participants are provided with [a file with a sample data format](https://github.com/ai-forever/memory_aij2025/blob/main/data/format_example.jsonl) which contains 4 unique entries. The main language of all data is Russian. 

#### Dataset description
Each entry contains:
1) a unique identifier — `id`;
2) a question — `question`;
3) the question type — `question_type`;
4) an answer — `ans`;
5) a set of sessions — `sessions`;
6) references to the session IDs where the answer can be found — `ans_session_ids`.

#### Dialogue
Each dialogue consists of a list of sessions, typically numbering several dozen.

#### Session
Each session has its own unique ID and is made up of a series of exchanges between the user (`user`) and the assistant (`assistant`).

#### Utterance
An utterance looks like this: `{"role": "role", "content": "text"}`.

#### Dataset example
In this example, we have translated the dialogues and questions from Russian into English for your convenience. Each line in the jsonl file contains an entry structured like this:
```
{
    "id": "1",
    "question": "What is my name?",
    "question_type": "fact_equal_session"
    "ans": "Ivan",
    "sessions": [
        {
            "id": "session_id1",
            "messages": [
                {"role": "user", "content": "Hello, my name is Ivan."},
                {"role": "assistant", "content": "Hello, nice to meet you, Ivan!"}
            ]
        },
        {
            "id": "session_id2",
            "messages": [
                {"role": "user", "content": "I have a cat named Barsik."},
                {"role": "assistant", "content": "Interesting! How old is Barsik?"},
                {"role": "user", "content": "He is 2."},
                {"role": "assistant", "content": "Oh, he’s still a kitten then."}
            ]
        },

        {
            "id": "session_id3",
            "messages": [
                {"role": "user", "content": "My dog Laika adores my girlfriend."},
                {"role": "assistant", "content": "It’s wonderful when your pets love the people close to you!"},
                {"role": "user", "content": "But my cat, on the contrary, is afraid of her."},
                {"role": "assistant", "Cats are generally less social animals, so there’s no need to worry too much about that."}
            ]
        },
        {
            "id": "session_id4",
            "messages": [
                {"role": "user", "content": "Remember I told you about my girlfriend?"},
                {"role": "assistant", "content": "I remember. Your dog really loves her."},
                {"role": "user", "content": "Well, now she’s not my girlfriend, but my wife."},
                {"role": "assistant", "You got married? Congratulations!"}
            ]
        }
    ],
    "ans_session_ids": ["session_id1"]
}
```

#### Questions
There are four types of questions:
| Type of question | Description | Example |
|-|-|-|
fact_equal_session | The answer to the question is contained within a single specific session | What’s my name? (Answer is in session_id1) |
info_consolidation | The answer requires aggregating information from multiple sessions. | What pets do I have? (Answer based on session_id2 and session_id3) |
info_updating | This type involves information about the user that changes over the course of the dialogue. | Am I married? (Answer based on session_id3 and session_id4; updated info appears in session 4) |
no_info  | Questions for which no answer exists in the dialogue. | How old am I? (No answer in the dialogue) |


#### Dialogue Specifics
* Users may write both short conversational phrases and longer requests, such as “summarize the article…” or “translate the text…”.
* Not all sessions contain user information.
* Some dialogues are taken from user interaction logs with GigaChat (personal data was anonymized), while others were generated by two models – one acting as the user and the other as the assistant. Some dialogues were created by foreign language models, so they may include elements and references common in Western culture.
* Questions can be asked about any session, not only the first or last one.

### Output Data
After successful inference, you should have a file named submit.csv, which will be used in the final solution evaluation. The set of unique identifiers in both input and output data must match.

The output file is a table with the following schema:

| id | answer | answer_time |
|-|-|-|
| 1 | Тебя зовут Василий. | 1.3924849033355713 |
| 2 | У тебя двое детей. | 0.7695510387420654 |

- **id** — unique identifier corresponding to the query in the input data;
- **answer** — the model’s response to the memory question based on the dialogue;
- **answer_time** — time taken to answer the question, in seconds.


## Metric calculation

- Metric type: Accuracy
- Meaning: The proportion of matching answers. The answer matches if it is semantically (in meaning) identical to the correct one
- Comparison technique: LLM as a judge

The evaluation script is provided with a submit.csv file with the answers to the test questions generated by the model and a file with the correct answers. Each pair of answers is sent to the LLM input, which, using detailed instructions, evaluates how similar the response of the submitter is to the correct answer to the question asked, and issues a binary verdict. If the attempt to obtain a verdict fails, the process is repeated k times until it is obtained. After k times, the verdict will be negative. The LLM generation parameters are fixed in such a way as to minimize the variability of the verdict, and, consequently, the final evaluation from launch to launch.

Important! Try to keep the answers in your submission short and informative, no more than 1 sentence.

You should not try to "cheat" the judge in your answers. Attempts at prompt injection, tricky wording, etc. are monitored. If such a submission gets into the top of the leaderboard, it will be excluded.

## Restrictions

A Participant or a Team of Participants can upload no more than 3 (three) solutions for evaluation during one day. Only valid attempts that have received a numerical evaluation are taken into account. If an exception occurs when calculating the metric, the solution is considered invalid, and the attempt counter is not reduced.

The container with the solution is launched under the following conditions:
* 243 Gb RAM
* 16 CPU-cores
* 1 GPU Tesla A100 (80 Gb)
* 10 GB of disk space
* Maximum time to complete the solution: 8 hours, including 7 hours to generate answers to questions and 1 hour to evaluate answers using "LLM as a judge"
* The environment in which the solution is launched does not have access to Internet resources
* Limit on the total weight of the downloaded solution: 5 Gb

## Baseline solution
As a baseline solution, participants are provided with [an implementation](https://github.com/ai-forever/memory_aij2025/tree/main/src/submit) that accumulates the entire dialogue context during the memorization stage. After that, at the question answering stage, the entire dialogue is submitted to the input of the model along with a test question. In the case where the dialogue length exceeds the maximum context length supported by the model, the question is sequentially asked only to the last phrases that fit into the context. The details of launching the baseline solution are described in the section ["Description of the solution format"](#description-of-the-solution-format).

## The prize pool
* First place – 900,000 rubles    
* Second place – 700,000 rubles     
* Third place – 400,000 rubles
