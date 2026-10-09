# Unit Two Capstone Project

A Python CLI application that answers questions using three coordinated agents in a multi-agent RAG model.

- **Quantitative questions** are answered by converting natural language to SQL and searching an SQL database.
- **Qualitative questions** are answered from documents via a RAG search with vectorized values.
- Each LLM response passes through a **validation layer**, while **token costs and usage** are logged per query.
- The model then returns an optimal answer to the user's query.

---

## System Architecture

> The architecture picture didn't transfer into the README, but it follows the structure implied in the assignment.

```text
CLI query --> Manager Agent --> Quantitative / Qualitative --> Validation Layer --> Tokenizer Layer
```

If the prompt is complex, the query goes back to the Manager Agent and out to another agent before reaching the validation and tokenizer layers.

---

## Setup & Usage

1. Install Gemini and Python on your machine.
2. Upload the code.
3. Install all required libraries.
4. Run the program with your desired query.

---

## Project Testing

For context, I developed a few fake files to use throughout testing and to answer the queries below. They contain information on a fake company's:

- Review process
- Customer complaints policy
- Customer success strategies guide
- Employee satisfaction policy
- Industry benchmark reference
- Sales performance policy
- Security policy

---

### Test One

I began development with the qualitative agent, so my first test was a simple summarization of the company's security policies.

**Query:** `What is the company's security policy`

**Result:**

**Access Management:** All workforce members must use company-managed accounts and multifactor authentication [Source 1]. Access reviews must be completed quarterly with a target of 100% completion within 10 business days of each quarter-end [Source 1]. Departing personnel must have their access removed within four hours of confirmed departure notification [Source 1].

**Data Classification:** Company data is classified as Public, Internal, Confidential, or Restricted [Source 1]. Customer contracts, credentials, employee records, and production data are Restricted and may be accessed only by authorized personnel with a documented business need [Source 1].

**Security Incidents:** Suspected security incidents must be reported to Security Operations immediately [Source 1]. Critical incidents require acknowledgement within 30 minutes, containment within four hours when practicable, and a documented post-incident review within 10 business days [Source 1].

**Engineering Controls:** Production changes require approved code review, automated testing, and deployment logging [Source 1]. Secrets must not be stored in source code [Source 1]. Vulnerability findings rated Critical or High must have an owner and remediation plan within two business days [Source 1].

**Training and Evidence:** Security training is required annually with a target completion rate of at least 98% [Source 1]. Required evidence includes access-review records, training reports, incident records, and deployment logs [Source 1].

**Governance:** Maya Chen, Director of Security Operations, is the policy owner and is responsible for policy monitoring and quarterly reporting [Source 1].

**Sources retrieved:**

| # | Source | Chunk | Distance |
|---|--------|-------|----------|
| 1 | `security_policy.txt` | 0 | 0.854 |
| 2 | `customer_complaints_policy.txt` | 0 | 1.337 |

**Validation Layer:** No flags

**Analysis:**

Many things are right with this response. The model correctly ran through the RAG process by identifying the correct files to source from and returning only information pertinent to the question. However, the retrieval process also drew from the customer complaints policy, which I don't believe is pertinent to the question, although it may share similarly vectorized key terms. As a result, I will set a limit in the retrieval process of a **max distance of 1**, then see if this fits with the other questions.

---

### Test Two

**Question:** `What is our customer's churn rate?`

**Output:**

**Route:** quantitative

**Answer:**

Your customer churn rate is **2.24%** (approximately 0.0224).

**Important Assumptions & Details:**

- **Calculation Method:** The churn rate was calculated by dividing the total number of churned customers by the total number of starting customers across the dataset (`SUM(churned_customers) / SUM(customers_start)`).
- **Data Limits:** The results were not truncated, and the output is based on a single summarized row.

**Analysis:**

This response was handled correctly: it first contacted the quantitative agent, which shows the Manager Agent is working as intended. Validation didn't flag anything.

---

### Test Three

*A case with a shifty output.*

**Question:** `How does our employee satisfaction compare to industry standards and what policies might impact this?`

**Answer:**

Based on the available documents, employee satisfaction is measured on a 0–100 scale through a quarterly anonymous survey, and the industry benchmark for employee satisfaction is 75 on the same scale. However, the provided context does not contain Northstar Analytics' actual current satisfaction score, so a direct comparison to the benchmark cannot be made.

**Policies and factors that may impact employee satisfaction:**

- **Employee Satisfaction Policy:** Satisfaction can be affected by factors such as code-review workload, customer escalation volume, sales target pressure, training requirements, and access to appropriate work tools. Additionally, improvement plans are required if scores fall below specific thresholds regarding workload, manager effectiveness, career development, tools, or operating processes.
- **Sales Performance Policy:** Sales performance and incentives can create pressures that impact satisfaction, and leadership is directed to review employee satisfaction alongside sales outcomes, support workload, and customer churn.

> **Note:** The quantitative analysis agent was unable to answer the question, as it was blocked from retrieving data from `business_metrics.csv`.

**Analysis:**

While the model was able to answer the question and even generated token usage between the models, this response can't be trusted, even though nothing was flagged at the validation level. The quantitative agent's contribution was missing, and it was, for some reason, not included in the search.

After a lot of testing, I realized this was because the question required a comparison between SQL data and a document, and I had set the quantitative agent to treat half-answered questions as unfinished, so it would simply say it wasn't used.

**Fix:** The quantitative agent now only reports "unanswerable" if *none* of the question can be solved using the data.