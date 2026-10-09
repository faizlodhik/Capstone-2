Unit Two Capstone Project:
Created a python CLI application that can answer questions using three coordinated
agents in a multi-agent RAG model. Quantitative questions are answered by converting
natural language to SQL and then searching through an SQL database. Qualitative
questions are answered from documents via an RAG search with vectorized values. Each of
these responses from the LLM is passed through a validation layer while token costs and
usages are logged per query. The model then returns an optimal answer to the user’s query.
System Architecture:
Picture didn't transfer in the readme. But follows the structure implicated in the assignment. CLI query --> Manager Agent --> Quantiative / Qualitative. If the prompt is complex then it would go back to manager and back out to another agent before going to the validation and tokenizer layers.
Setup & Usage:
The setup simply involves working through installing gemini / python on your machine. Uploading the code, installing all libraries, and then running the program with your desired query.
Project Testing:
For context I developed a few faux files to be used through the testing process and to
answer the queries provided at the end. These files contain information on a fake
company’s review process, customer complaints policy, customer success strategies
guide, employee satisfaction policy, industry benchmark reference, sales performance
policy, and security policy.
Test One:
I first began my development with the qualitative agent and so my first test was a simple
summarization of the companies security policies
Query: “What is the company’s security policy”
Result:
Access Management:* All workforce members must use company-managed accounts
and multifactor authentication [Source 1]. Access reviews must be completed quarterly
with a target of 100% completion within 10 business days of each quarter-end [Source 1].
Departing personnel must have their access removed within four hours of confirmed
departure notification [Source 1].
Data Classification: Company data is classified as Public, Internal, Confidential, or
Restricted [Source 1]. Customer contracts, credentials, employee records, and production
data are Restricted and may be accessed only by authorized personnel with a documented
business need [Source 1].
Security Incidents: Suspected security incidents must be reported to Security
Operations immediately [Source 1]. Critical incidents require acknowledgement within 30
minutes, containment within four hours when practicable, and a documented post-
incident review within 10 business days [Source 1].
Engineering Controls: Production changes require approved code review, automated
testing, and deployment logging [Source 1]. Secrets must not be stored in source code
[Source 1]. Vulnerability findings rated Critical or High must have an owner and remediation
plan within two business days [Source 1].
Training and Evidence: Security training is required annually with a target completion
rate of at least 98% [Source 1]. Required evidence includes access-review records, training
reports, incident records, and deployment logs [Source 1].
Governance: Maya Chen, Director of Security Operations, is the policy owner and is
responsible for policy monitoring and quarterly reporting [Source 1].
Sources retrieved:
[1] security_policy.txt (chunk 0, distance 0.854)
[2] customer_complaints_policy.txt (chunk 0, distance 1.337)
Validation Layer: No flags
When analyzing this response, many things are right. The model was able to correctly run
through the RAG process by identifying the correct files to source from and returning only
pertinent information to the question. However, we can see that during the retrieval policy
process it also drew from customer complaints policy, which I believe isn’t pertinent to the
question although it may also have some similarly vectorized key terms. As a result, I will
set a limitation in the retrieval process to a max distance of 1. And then see if this fits right
with the other questions.
Test two:
Question: What is our customer’s churn rate?
Output:
Route: quantitative
Answer:
Your customer churn rate is 2.24% (approximately 0.0224).
Important Assumptions & Details:
Calculation Method: The churn rate was calculated by dividing the total number of
churned customers by the total number of starting customers across the dataset
(`SUM(churned_customers) / SUM(customers_start)`).
Data Limits: The results were not truncated, and the output is based on a single
summarized row.
This response was conducted in the correct way, by first contacting the quantitative model
it shows that the manager agent is working correctly. Validation didn’t flag anything.
Test Three, a case with a shifty output:
Question: How does our employee satisfaction compare to industry standards and what
policies might impact this?
Answer: Based on the available documents, employee satisfaction is measured on a 0–100
scale through a quarterly anonymous survey, and the industry benchmark for employee
satisfaction is 75 on the same scale. However, the provided context does not contain
Northstar Analytics' actual current satisfaction score, so a direct comparison to the
benchmark cannot be made.
Policies and factors that may impact employee satisfaction:
Employee Satisfaction Policy: Satisfaction can be affected by factors such as code-
review workload, customer escalation volume, sales target pressure, training
requirements, and access to appropriate work tools. Additionally, improvement plans are
required if scores fall below specific thresholds regarding workload, manager
effectiveness, career development, tools, or operating processes.
Sales Performance Policy: Sales performance and incentives can create pressures
that impact satisfaction, and leadership is directed to review employee satisfaction
alongside sales outcomes, support workload, and customer churn.
Note: The quantitative analysis agent was unable to answer the question, as it was
blocked from retrieving data from `business_metrics.csv`.
While the model was able to answer the question and even generated token usage between
the models, this response can’t be trusted although nothing was flagged at the validation
level. I can’t trust this output due to the lack of reference from the quantitative model
which was for some reason not included in the search. After a lot of testing, I realized that
this is due to the fact the question required a comparison between SQL and a document
and the quantitative model I had just left half-answered questions as unfinished so it would
just say it wasn’t used. Now it only says unanswerable if none of the question can be solved
using the data.