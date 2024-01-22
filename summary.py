import torch
import transformers
from transformers import AutoTokenizer
from langchain.chains import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, low_cpu_mem_usage=True)

print("Generating pipeline")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    max_length=3000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

print("Pipeline is ready")
print(pipeline)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

template = """
              Write a summary of the following text delimited by triple backticks.
              Return your response which covers the key points of the text.
              ```{text}```
              SUMMARY:
           """

prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

print("LLM is ready")
print(llm_chain)

text = """
Patient Name: John Doe
Date of Birth: January 15, 1975
Gender: Male
Medical Record Number: 123456789
Date of Assessment: August 18, 2023

I. Chief Complaint:
The patient presents with complaints of persistent fatigue, unexplained weight loss, and intermittent abdominal pain over the past few months. He reports a gradual decrease in appetite and occasional nausea. The patient is seeking medical evaluation to determine the underlying cause of his symptoms.

II. Medical History:
The patient has a history of hypertension managed with medication for the past five years. He underwent an appendectomy in his late twenties and had a hernia repair surgery a decade ago. The patient reports a family history of diabetes on his maternal side.

III. Review of Systems:

General: The patient reports fatigue, unexplained weight loss of approximately 10 pounds over three months, and a decreased appetite.

Gastrointestinal: The patient experiences intermittent abdominal pain, predominantly in the right upper quadrant, without a clear trigger. He reports occasional nausea, and denies vomiting, diarrhea, or changes in bowel habits.

Cardiovascular: The patient's blood pressure has been well controlled with medication. He denies chest pain, palpitations, or shortness of breath.

Respiratory: The patient denies cough, wheezing, or shortness of breath.

Musculoskeletal: No significant joint pain or limitations in mobility reported.

Neurological: The patient denies headaches, dizziness, or changes in vision.

Psychological: The patient mentions occasional stress due to work-related factors but denies symptoms of depression or anxiety.

IV. Physical Examination:

Vital Signs: Blood pressure is 130/80 mmHg, heart rate is 78 beats per minute, respiratory rate is 16 breaths per minute, and temperature is 98.6°F (37°C).

General: The patient appears fatigued but alert and oriented to person, place, and time. He appears to have lost weight since his last visit.

Abdominal Examination: There is tenderness on palpation in the right upper quadrant of the abdomen. No palpable masses or organomegaly noted. Bowel sounds are normal.

Cardiovascular Examination: Regular rate and rhythm with no murmurs or abnormal sounds.

Respiratory Examination: Clear breath sounds bilaterally, no wheezing or crackles noted.

Neurological Examination: No focal neurological deficits observed.
"""

print("Running llm_chain")
print(llm_chain.run({"text": text}))
print("Done running llm_chain")
