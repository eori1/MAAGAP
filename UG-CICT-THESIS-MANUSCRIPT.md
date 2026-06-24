# MAAGAP: A Machine Learning Framework for Predictive Risk Assessment and Optimized Resource Allocation in Government Project Management

An Undergraduate Thesis

Presented to the Faculty of the

College of Information and Communications Technology

West Visayas State University

La Paz, Iloilo City

In Partial Fulfillment

of the Requirements for the Degree

Bachelor of Science in Computer Science

by

Jullian A. Bilan

Clarence Anthony G. Bolivar

Kirk Henrich C. Gamo

Jan Floyd J. Vallota

June 2027

---

## Approval Sheet

MAAGAP: A Machine Learning Framework for Predictive Risk Assessment and Optimized Resource Allocation in Government Project Management

An Undergraduate Thesis for the Degree

Bachelor of Science in Computer Science

by

Jullian A. Bilan

Clarence Anthony G. Bolivar

Kirk Henrich C. Gamo

Jan Floyd J. Vallota

| Panel | | Panel |
|-------|-|-------|
| | | |
| | | |
| Panel | | Panel |
| | | |
| | | John Cristopher Mateo |
| Panel | | Adviser |

Concurred:

Name Here Dr. Ma. Beth S. Concepcion

Dept./Div. Chair, [Div/Dept.] Dean

June 2027

---

## Acknowledgment

[Example Content - The researchers would like to express their deepest appreciation to the following persons, who in one way or another have made this work possible:]

[Some content here]

Jullian A. Bilan

Clarence Anthony G. Bolivar

Kirk Henrich C. Gamo

Jan Floyd J. Vallota

June 2027

[A. Lastname]; [B. Lastname], "Title Here," Unpublished Undergraduate Thesis, [Degree Program], College of Information and Communications Technology, West Visayas State University, Iloilo City, Philippines, 2026.

---

## Abstract

[Abstract should not be more than 250 words.]

**Keywords:** keyword1, keyword2, keyword3

---

## Table of Contents

- Approval Sheet
- Acknowledgment
- Abstract
- Table of Contents
- List of Figures
- List of Tables
- List of Appendices
- CHAPTER 1 INTRODUCTION TO THE STUDY
  - Background of the Study and Conceptual Framework
  - Significance of the Study
  - Delimitation of the Study
  - Definition of Terms
- CHAPTER 2 REVIEW OF RELATED STUDIES
  - Review of Existing and Related Studies
  - Current Systems
  - Related Systems or Solutions
  - Related Studies
- CHAPTER 3 RESEARCH DESIGN AND METHODOLOGY
  - Description of the Proposed Study
  - Methods and Proposed Enhancements
  - Components and Design
  - System Architecture
  - Software Architecture
  - Database Design
  - Procedural Design
  - Object-Oriented Design
  - Process Design
  - Methodology
  - System Development Life Cycle
- References
- Appendices

---

## List of Figures

- Figure 2 MAAGAP LSTM Architecture
- Figure 3 MAAGAP SYSTEM ARCHITECTURE
- Figure 4 MAAGAP SOFTWARE ARCHITECTURE
- Figure 5 MAAGAP ENTITY-RELATIONSHIP DIAGRAM (ERD)
- Figure 6 MAAGAP PROCEDURAL DESIGN AND PROCESS FLOWCHART
- Figure 7 MAAGAP USE CASE DIAGRAM
- Figure 8 MAAGAP CLASS DIAGRAM
- Figure 9 MAAGAP STATE MACHINE DIAGRAM
- Figure 10 MAAGAP DEPLOYMENT DIAGRAM
- Figure 11 MAAGAP DATA FLOW DIAGRAM LEVEL 0
- Figure 12 MAAGAP DATA FLOW DIAGRAM LEVEL 1
- Figure 13 MAAGAP DATA FLOW DIAGRAM LEVEL 2 (PREDICTION ENGINE)
- Figure 14 MAAGAP DATA FLOW DIAGRAM LEVEL 2 (OPTIMIZATION ENGINE)
- Figure 15 ITERATIVE SDLC PROCESS

---

## List of Tables

- Table II COMPARATIVE ANALYSIS OF COMMERCIAL AI-DRIVEN PREDICTIVE SYSTEMS AND MAAGAP
- Table III EMPIRICAL EVALUATION RESULTS OF MACHINE LEARNING ALGORITHMS IN RELATED STUDIES

---

## List of Appendices

- Appendix A Sample Appendix
- Appendix B Disclaimer

---

# CHAPTER 1 INTRODUCTION TO THE STUDY

## Background of the Study and Conceptual Framework

Government infrastructure projects consistently face critical challenges, including schedule delays, budget overruns, and inefficient resource allocation that undermine public trust and limit service delivery effectiveness. Despite advances in project monitoring technology, current systems remain fundamentally reactive, tracking what has already happened rather than anticipating future problems. This gap is particularly acute in the Philippines, where infrastructure projects experience significant delays due to right-of-way acquisition issues, contractor disputes, funding bottlenecks, and coordination challenges across multiple government agencies [75]. The Commission on Audit (COA) consistently reports project implementation delays, unutilized appropriations, and suboptimal resource deployment across national agencies and local government units [15]. Specific to the 2024 cycle, 747 infrastructure projects under the DPWH were flagged as "unusable, idle, or incomplete" [55]. Similar budget execution challenges were observed in the DOH, which returned approximately ₱800 million in unspent funds [24], while ₱405.5 million in medical equipment under the Health Facilities Enhancement Program (HFEP) remained unutilized by year-end [45].

The problem intensifies when government resources are distributed suboptimally across concurrent projects, with some receiving excess allocation while others face critical shortages. This inefficiency becomes especially problematic during disaster response scenarios in the Philippines, where emergency reconstruction projects compete for limited resources while facing urgent timelines. The country's exposure to typhoons, earthquakes, and flooding adds another layer of uncertainty to project planning [5]. Current monitoring systems answer what happened and what is happening, but rarely address the more valuable questions: what will happen, and what should we do about it? Recent advances in anomaly detection and fraud identification focus primarily on identifying problems after they occur rather than preventing them [69].

The shift toward Industry 4.0 has driven a transition from manual project monitoring to predictive analytics. Traditional methodologies often fail to account for the stochastic nature of large-scale infrastructure, leading to the systemic delays reported by oversight bodies [56]. To address this, modern research leverages machine learning (ML) and optimization algorithms---such as Support Vector Regression (SVR) and Artificial Neural Networks (ANN)---to forecast success and identify failure triggers [46]. Recent Philippine studies indicate that hybrid models incorporating Principal Component Analysis (PCA) can achieve predictive accuracies exceeding 85% [56,37]. Despite these advancements, local implementation remains in its nascent stages, frequently relying on retrospective audits rather than real-time, ML-based risk mitigation [44].

This proposal introduces MAAGAP (Machine Analytics for Allocation, Governance and Assessment of Projects), an AI-driven system that transforms government project management from reactive troubleshooting to proactive risk mitigation. MAAGAP addresses these challenges through an integrated platform built on three foundational components. The Predictive Analytics Module employs ensemble machine learning models---including Random Forest, Gradient Boosting, and LSTM networks---trained on historical project data to forecast delays and cost overruns. The Risk Scoring Module classifies projects into actionable risk categories (low, medium, high, critical), triggering automated alerts and tailored interventions. The Optimization Module formulates resource allocation as a constrained optimization problem, generating concrete recommendations that maximize project success probability while respecting budget, manpower, and equipment limitations.

The system's early warning capabilities identify at-risk projects weeks or months in advance, providing sufficient time for preventive action. An interactive dashboard presents real-time risk heatmaps, timeline projections, and resource utilization charts in accessible formats for decision-makers. Importantly, MAAGAP incorporates explainable AI principles, providing clear reasoning behind predictions and recommendations to support accountability and build trust. For agencies managing geographically distributed projects, MAAGAP offers a centralized view of risk across the entire portfolio, enabling strategic intervention before problems escalate.

## Objectives of the Study

This study aims to develop and validate MAAGAP, an AI-driven predictive risk assessment and resource optimization system that transforms government project management from reactive monitoring to proactive risk mitigation through integrated machine learning and optimization algorithms.

Specifically, it aims to:

1. Develop and train a multi-stage predictive framework by implementing Random Forest and Gradient Boosting (XGBoost) classifiers for feature-based risk categorization, integrated with Long Short-Term Memory (LSTM) networks to capture temporal dependencies in sequential monitoring data, utilizing preprocessed historical timelines, contractor records, and external contextual variables---such as PAGASA weather patterns and PSA economic indicators---to forecast project delays and cost overruns.

2. Evaluate the performance of the trained models using a 70/30 train-test split on historical and contextual datasets by applying standard classification and regression metrics---specifically Accuracy, Precision, Recall, F1-Score, and AUC-ROC---to assess the discriminative ability of risk categories, alongside Mean Absolute Error (MAE) to quantify the average prediction error in terms of project delays and cost overruns.

3. Design a dynamic risk scoring engine that translates probability outputs from the predictive models into actionable tiers "Low, Medium, High, and Critical" by applying logic consistency testing and defined threshold boundaries to ensure the reliability of risk classifications.

4. Develop and evaluate optimization algorithms using linear programming to generate resource reallocation recommendations under constraints such as budget, manpower, and equipment. Success will be measured by demonstrating at least a 15% improvement in allocation efficiency compared to current manual approaches through simulated project scenarios.

5. Develop and evaluate optimization algorithms using linear programming to generate resource reallocation recommendations under constraints such as budget, manpower, and equipment. Success will be measured by demonstrating at least a 15% improvement in allocation efficiency compared to current manual approaches through simulated project scenarios.

6. Evaluate the prototype's software quality using the ISO/IEC 25010 standard, focusing on sub-characteristics such as Functional Suitability, Usability, and Reliability.

## Significance of the Study

The MAAGAP research framework provides measurable value across the entire ecosystem of public service, from high-level fiscal policy to the granular management of construction sites and the broader academic pursuit of smarter governance.

The results of this study will be useful to the following:

**National Government Agencies (DBM, DPWH, BTr, and COA).** This study provides a mechanism for ensuring fiscal discipline and resilience by predicting overruns and delays, thereby optimizing the budget execution phase. It addresses the systemic issue of unutilized allotments frequently flagged by oversight bodies and strengthens accountability through transparent, data-driven rationale for resource shifts [26,74]. Furthermore, it assists in the rapid reallocation of emergency funds and equipment in disaster-prone regions, improving national crisis response capabilities [57].

**Local Government Units (LGUs) and Provincial Oversight Offices.** The research offers a practical application of "Smart Government" by aligning risk scoring with current provincial oversight protocols. By integrating logistical constraints and personnel capacity into the optimization engine, the study provides LGUs with a realistic framework for managing local infrastructure and non-infrastructure projects within resource-constrained environments.

**Project Managers and Field Inspectors.** This study serves as a Decision Support System (DSS) that augments human judgment with high-speed data processing. It allows managers to transition from manual "firefighting" to proactive risk treatment strategies by providing early warnings on high-risk bottlenecks [57,58]. The optimization module offers actionable recommendations for the deployment of limited personnel and machinery, shifting the focus from routine rescheduling to strategic leadership [29].

**Public Administration Practitioners and Policymakers.** The findings contribute to the digital transformation of the public sector by providing empirical evidence on which project features strongest predict success in a developing-nation context [71]. It establishes best practices for implementing AI-driven tools in bureaucratic settings, offering a template for navigating the intersection of social science, artificial intelligence, and operations research [74].

**Future Researchers.** This study serves as a foundational blueprint for scholars exploring the integration of emergent technologies in high-stakes governance environments [4]. It contributes to the evolving field of data-driven project management by providing a validated framework for embedding predictive analytics into traditional methodologies [3]. By documenting implementation challenges and technical gaps, the research identifies critical areas for future academic inquiry into the refinement of public sector machine learning applications [70,2].

## Delimitation of the Study

To ensure feasibility and technical focus within the research timeline, the study is bounded by specific constraints regarding its project types, technical methodologies, and data parameters.

The research focuses specifically on government infrastructure and non-infrastructure projects---including roads, bridges, public buildings, and healthcare facilities---at the local government unit (LGU) level, with a geographic emphasis on Iloilo Province. Utilizing datasets from the Provincial Planning and Development Office (PPDO), the study ingests contractor records, inventory, and historical logs, while assuming an "ideal setup" for initial data modeling to facilitate development. The analysis is strictly delimited to two project categories with predefined standard durations: six months for non-infrastructure programs and one year for infrastructure projects. The predictive engine specifically targets "Red Flags," defined as completion rates exceeding these durations without filed extensions due to environmental or weather-related factors.

Risk scoring is delimited to Low, Medium, and High categories to align with PPDO oversight protocols, while the optimization engine is constrained by the real-world capacity of five to six permanent inspectors. Recommendations are further bounded by logistical hurdles, such as vehicle availability and administrative paperwork loads, which impact inspection frequency. Technical visuals are limited to Actual Time vs. Scheduled Time deviations and status graphs for projects classified as Ongoing, Monitored, or Completed. Finally, the research specifically excludes the modification of the existing PPDO Management Information System (MIS), positioning the proposed system solely as a complementary decision-support tool, and excludes highly specialized projects, such as classified defense or aerospace infrastructure, governed by distinct security protocols.

The study is delimited to the use of specific, proven machine learning and optimization techniques:

- **Predictive Modeling:** While various ML architectures exist, this study focuses on evaluating the comparative performance of Random Forest, Gradient Boosting, and LSTM within the specific context of Philippine public infrastructure---a domain where their combined predictive accuracy has yet to be fully explored.
- **Optimization:** Resource allocation will be modeled as a linear programming problem and solved using the PuLP library. The research does not explore non-linear or heuristic optimization methods, prioritizing a mathematically rigorous approach that fits within standard LGU computing constraints.
- **User Interface:** The prototype will be developed as a web-based dashboard using Streamlit or Flask, chosen for their rapid prototyping capabilities rather than production-grade, enterprise-scale security features.

The research relies on historical project data from 2016 to 2025. Key limitations include:

- **Data Availability:** Should partner agencies have incomplete records, the study will utilize synthetic data generation, which may not perfectly capture all real-world "edge cases" or contractor-specific nuances.
- **Evaluation Period:** The 8--10 month research timeline restricts the ability to evaluate long-term system adoption or the "Continuous Learning" outcomes of the models over multiple fiscal cycles.
- **Focus:** The study concentrates on delay and cost overrun prediction, excluding other important but secondary metrics such as post-occupancy facility quality or environmental sustainability scores.

## Definition of Terms

For better understanding, the following terms were defined conceptually and operationally:

**Accuracy** -- A performance metric calculated as the ratio of correct predictions to the total number of input samples.

**Accuracy** -- In this study, this refers to the metric used to validate the reliability of the forecasting engine in identifying project outcomes.

**Allotment** -- An authorization issued by the DBM to an agency permitting it to commit/incur obligations and/or pay out funds within a specified period.

**Allotment** -- In this study, this refers to the specific budgetary ceiling within which an agency must operate for its documented projects.

**Budget Execution** -- The phase in the national budget cycle where funds are released and projects are implemented.

**Budget Execution** -- In this study, this refers to the operational stage involving the issuance of Special Allotment Release Orders (SARO) and Notices of Cash Allocation (NCA).

**Explainable AI (XAI)** -- A set of techniques applied during the machine learning lifecycle to make AI outputs understandable and transparent to human users.

**Explainable AI (XAI)** -- In this study, this refers to the methods used to support accountability by providing clear justifications for government decision-making suggestions.

**F1-Score** -- The harmonic mean of precision and recall.

**F1-Score** -- In this study, this refers to the evaluation metric used to balance "False Alarms" and "Missed Dangers" within the risk models.

**Internal Control** -- The organization-wide plan and all adopting methods/measures to safeguard assets and ensure adherence to laws and regulations.

**Internal Control** -- In this study, this refers to the compliance framework involving the NGICS and PGIAM to ensure accurate accounting and legal adherence.

**Linear Programming (LP)** -- A mathematical optimization technique used to find the best possible outcome in a model whose requirements are represented by linear relationships.

**Linear Programming (LP)** -- In this study, this refers to the mathematical approach used to determine the most efficient distribution of project resources.

**LSTM (Long Short-Term Memory)** -- A recurrent neural network architecture designed to learn and remember long-term dependencies.

**LSTM (Long Short-Term Memory)** -- In this study, this refers to the specific machine learning model applied to analyze the sequential nature of project lifecycles.

**Negative Slippage** -- A technical project status indicating that the actual physical progress of a project is lagging behind scheduled progress.

**Negative Slippage** -- In this study, this refers to the percentage-based lag used as a key indicator of project underperformance.

**Notice of Cash Allocation (NCA)** -- A document issued by the DBM to government servicing banks specifying the maximum cash withdrawal amount for an agency.

**Notice of Cash Allocation (NCA)** -- In this study, this refers to the actual liquid funds available to an agency to settle its project-related obligations.

**Optimized Resource Allocation (ML Context)** -- The strategic, data-driven management of resources that dynamically adjusts to real-time conditions.

**Optimized Resource Allocation (ML Context)** -- In this study, this refers to the machine learning-assisted distribution of budget and personnel to maximize output while minimizing costs.

**Predictive Risk Assessment** -- A proactive methodology that uses historical data and machine learning to anticipate and mitigate potential threats.

**Predictive Risk Assessment** -- In this study, this refers to the automated process used to identify project delays or cost breaches before they occur.

**Resource Allocation** -- The critical exercise of distributing revenues and borrowed funds to attain economic and social goals.

**Resource Allocation** -- In this study, this refers to the distribution process outlined by the DBM and NEDA for Philippine government projects.

**Risk Scoring** -- A quantification process where predictions are translated into discrete categories based on probability thresholds.

**Risk Scoring** -- In this study, this refers to the categorization of projects from "Low" to "Critical" to prioritize administrative intervention.

---

# CHAPTER 2 REVIEW OF RELATED STUDIES

## Review of Existing and Related Studies

### Current Systems

#### 2.1 Current Project Monitoring and Management Systems in the Philippine Government

The Philippine government has implemented various systems to monitor and manage infrastructure projects, yet significant challenges persist in ensuring timely delivery, budget compliance, and effective resource allocation. Understanding these existing systems provides essential context for identifying the gaps that MAAGAP aims to address.

Collectively, existing Philippine government systems emphasize transparency, compliance, and real-time monitoring. While these platforms improve public access to infrastructure data, they remain reactive in nature. None integrate predictive risk modeling, delay forecasting algorithms, or prescriptive decision-support mechanisms for optimizing project portfolios. This systemic orientation toward disclosure rather than anticipatory intelligence highlights the research gap addressed by MAAGAP.

##### 2.1.1 DPWH Transparency Portal

The most recent and significant development in Philippine infrastructure monitoring is the DPWH Transparency Portal, launched by President Ferdinand R. Marcos Jr. in November 2025 [50]. The portal was introduced as part of the government's broader transparency initiative aimed at addressing corruption and improving public accountability in infrastructure spending, this AI-powered platform represents the government's strongest reform measure following months of revelations regarding ghost projects, kickbacks, and anomalies in flood control projects [22].

The portal provides citizens with real-time access to data on government construction projects nationwide, containing information on over 247,172 DPWH projects from 2016 to 2025 [50]. Through the website (https://transparency.dpwh.gov.ph/), users can view specific project details including location data, procurement documents, contractor contact information, implementing offices, budgets, accomplishments, target dates, and current status updates [22].

A notable feature is the integration of geotagged photographs documenting project progress from pre-construction to completion, supplemented by satellite images supplied by the Philippine Space Agency and private partners [22]. This visual documentation enables citizens to verify whether reported accomplishments match actual on-the-ground progress. The portal also incorporates procurement livestreams, including real-time bidding proceedings, further enhancing transparency in the contracting process [22].

The platform functions as "Sumbong sa Pangulo 2.0," an expanded version of the earlier complaint website that focused primarily on flood control projects [32]. Presidential Communications Office Acting Secretary Dave Gomez emphasized that users can search for any DPWH project and view critical details, with the system designed to be highly user-friendly: "Sa search bar, i-type mo lang kung ano yung proyekto gusto mong hanapin---lalabas na lahat ng detalye" [32].

To make searches even more accessible, the portal includes a conversational AI assistant available in both Filipino and English, designed so that even non-technical users can navigate it easily [22]. Users may tag projects with status indicators such as completed, defective, duplicate, unfinished, or ghost, creating a feedback mechanism that enables public participation in project monitoring [22].

President Marcos framed the portal as fulfilling the constitutional right to information: "Ginagawa natin full disclosure dahil karapatan malaman ng lahat kung paano ginagastos ang pera ng bayan" [22]. He described transparency as "the best medicine" for corruption, emphasizing that "open everything up to the sunlight because the people need to know" [22].

Despite these advances, the DPWH Transparency Portal remains fundamentally a transparency and monitoring tool rather than a predictive or prescriptive system. While its AI capabilities enable search and user interaction, the system does not forecast potential delays, identify at-risk projects before problems manifest, or recommend resource allocation adjustments. It answers questions about what is happening and what has happened, but not what will happen or what should be done about it---precisely the gap MAAGAP addresses.

##### 2.1.2 Project DIME (Digital Information for Monitoring and Evaluation)

Another significant initiative is the Digital Information for Monitoring and Evaluation (Project DIME) of the Department of Budget and Management (DBM). Launched in May 2025, Project DIME serves as a digital monitoring platform designed to provide real-time data on the status of big-ticket infrastructure programs [23]. The system displays key information including completion rates, budget sources, implementing agencies, and contractor details, making project data more accessible to both government stakeholders and the public [35].

Project DIME represents a notable step toward transparency and accountability in public infrastructure spending. As Budget Secretary Amenah Pangandaman articulated, the system responds to citizens' growing demand for openness, enabling them to "rightly demand for greater openness, transparency, and accountability, especially for projects and initiatives that have significant impact in their lives" [23]. The platform aligns with President Marcos Jr.'s Executive Order No. 31, series of 2023, which institutionalized the Philippine Open Government Partnership (PH-OGP) and committed the Philippines to promoting transparency and leveraging technology in governance [23].

A significant enhancement to Project DIME emerged through a memorandum of understanding signed in May 2025 between DBM, the University of the Philippines Nationwide Operational Assessment of Hazards (UP-NOAH) Center, and the DPWH [17]. This collaboration aims to integrate disaster risk reduction strategies into the system, ensuring that government projects are evaluated for resilience against natural hazards [17]. UP-NOAH's involvement brings expertise in hazard assessment, enabling the platform to incorporate environmental risk factors that could affect project implementation and longevity [60].

In August 2025, the system was further expanded with the launch of its Flood Control Project Component, allowing citizens to monitor flood control projects in real-time through geotagging and satellite imagery [21]. Budget Secretary Pangandaman emphasized that this component enables public participation, stating, "Puwede-puwede po kayong mag-log, magbigay ng komento, magbigay ng input doon sa website, sa system" (you can log in, give comments, and provide input on the website and system) [21].

Like the DPWH Transparency Portal, Project DIME focuses on monitoring and transparency rather than prediction and optimization. It displays what is happening and what has happened but does not provide anticipatory analytics or prescriptive recommendations.

##### 2.1.3 PhilGEPS (Philippine Government Electronic Procurement System)

The Philippine Government Electronic Procurement System (PhilGEPS) serves as the single, centralized electronic portal that is the primary and definitive source of information on government procurement [59]. Operating under the Procurement Service -- Department of Budget and Management (PS-DBM), PhilGEPS has undergone modernization to support the government's digital transformation efforts.

Under the New Government Procurement Act (NGPA), PhilGEPS now undertakes the role of being the primary channel in the conduct of government procurement activities, employing new features geared towards the fulfillment of a complete government online procurement process that improves analytics insight and system integration [59]. The modernized system includes several key platforms:

**The Virtual Store and eMarketplace:** Launched in December 2024, the PhilGEPS eMarketplace enables government agencies to directly purchase common-use supplies and equipment from competent and reputable suppliers through an interface similar to commercial e-commerce platforms [33]. Budget Secretary Amenah Pangandaman explained: "With the eMarketplace, government agencies or procuring entities can now just 'Add to Cart' or directly purchase their common-use supplies and equipment (CSE) requirements... With only a few clicks, we can now purchase the same way we would shop in Shopee or Lazada using our digital devices, shortening the tedious process of regular procurement from three months to just 60 days" [33].

The eMarketplace is designed to be inclusive, offering opportunities for micro, small and medium enterprises (MSMEs), social enterprises, and women-led businesses to participate in government procurement by simplifying the registration and bidding process [33]. It also helps combat corruption by verifying merchants and suppliers, ensuring they meet the technical specifications and budget requirements set by procuring entities [33].

**The eBidding Facility:** PhilGEPS supports competitive public bidding through its electronic procurement platforms, enabling transparent and efficient conduct of procurement activities [59]. The system publishes procurement programs and projects with detailed information including implementing offices, modes of procurement, schedules for each procurement activity, sources of funds, and estimated budgets [52].

PS-DBM Deputy Executive Director Philip Josef Vera Cruz noted that the launch of the eMarketplace is "just the beginning" of the ongoing journey of procurement reform, guided by the provisions of the New Government Procurement Act [33]. The system represents a significant advance in procurement transparency and efficiency, but like other Philippine government systems, it focuses on transaction processing and monitoring rather than predictive analytics for project outcomes.

##### 2.1.4 Commission on Audit Programs/Projects/Activities Disclosure

The Commission on Audit (COA) has established guidelines requiring all government agencies to practice full disclosure of their programs, projects, and activities (PPAs) [16]. These guidelines are grounded in constitutional mandates: Section 28, Article II of the 1987 Philippine Constitution adopts a policy of full disclosure of all transactions involving public interest, while Section 7, Article III recognizes the right of the people to information on matters of public concern [16].

Under COA guidelines, at the beginning of each year, all government agencies must provide their assigned Supervising Auditors and Audit Team Leaders with a list of all ongoing PPAs and those to be implemented during the year. The list must include [16]:

- Project name
- Implementing unit, office, or division
- Brief description of the PPA
- Contractor or supplier (if any)
- Mode of procurement
- Funding source
- Cost or approved budget
- Project duration including start and completion dates
- Location

These disclosure requirements are designed to ensure that government resources are "managed, expanded or utilized in accordance with law and regulations, and safeguarded against loss or wastage through illegal or improper disposition, with a view to ensuring efficiency, economy and effectiveness in the operations of government" [16]. The responsibility for faithful adherence to these principles rests directly with the chief or head of each government agency.

COA's authority to promulgate these auditing and accounting rules derives from Section 2(2), Article IX-D of the 1987 Constitution, which grants the Commission exclusive authority to establish rules "for the prevention and disallowance of irregular, unnecessary, excessive, extravagant, or unconscionable expenditures, or uses of government funds and properties" [16].

While COA's disclosure framework establishes important transparency requirements, it remains a compliance and auditing mechanism rather than a predictive system. The information collected supports retrospective audit activities---identifying irregularities after expenditures have occurred---rather than forecasting future project risks or optimizing resource allocation across project portfolios.

##### 2.1.5 PhilSA-DPWH Space Data Integration

A complementary initiative involves the Philippine Space Agency (PhilSA) and DPWH, which formalized a partnership in November 2025 to utilize space data for infrastructure monitoring [53]. Under this agreement, PhilSA provides DPWH with satellite imagery and space-based analytics to track and assess infrastructure developments across the country. The collaboration leverages PhilSA's archives, open-access satellite data, and commercial satellite subscriptions to provide comprehensive historical and up-to-date views of projects, enabling progress validation "from space" [53].

PhilSA Officer-in-Charge Gay Jane P. Perez emphasized the partnership's significance for governance, stating that satellite imagery makes "information on infrastructure projects more accessible and transparent to the public" and empowers "citizens to see progress on the ground from the vantage point of space" [53]. Both agencies have committed to developing advanced monitoring tools that integrate satellite remote sensing, artificial intelligence, and geospatial analytics to enable evidence-based project monitoring.

This initiative demonstrates the Philippine government's growing recognition of advanced technologies' potential in project oversight. The integration of AI mentioned in the agreement suggests future potential for predictive capabilities, but current implementations remain oriented toward retrospective and real-time monitoring rather than anticipatory analytics [53].

##### 2.1.6 Reliance on Manual and Semi-Automated Tools in Government Project Management

Beyond the flagship transparency and procurement systems, a significant portion of project monitoring and data management in Philippine government agencies continues to rely on manual processes and basic office software. This reliance on tools such as Microsoft Excel, Google Sheets, and traditional filing systems represents an important dimension of the current technological landscape that MAAGAP must consider.

**Training Initiatives for Spreadsheet-Based Data Management**

Government agencies actively conduct training programs to build personnel capacity in using spreadsheet applications for data management, indicating the prevalence of these tools in daily operations. In April 2024, the Philippine Statistical Research and Training Institute (PSRTI) conducted a three-day training on "Mastering Data Management: A Guide to Google Sheets and MS Excel" for members of the Regional Statistics Committee-BARMM [6]. The training was offered in response to requests from Regional Statistics Committees nationwide, highlighting the widespread need for proficiency in these basic tools. According to PSRTI, the training aimed to capacitate participants on "effective data management, analysis, and presentation" and enabled trainees to "create and format spreadsheets, learn to sort and filter, create formulas, use a range of functions, and get to work with pivot tables, charts, and other functionalities" [6].

Similarly, in August 2025, PSRTI conducted a four-day customized training course on Data Management for officials and staff of the Sandiganbayan, equipping 50 participants with "practical skills in organizing, analyzing, and managing data efficiently using MS Excel and Google Sheets" [54]. The training was delivered through lectures and hands-on exercises, reflecting the practical importance of these tools in judicial administrative functions.

These training initiatives demonstrate that despite the existence of specialized government information systems, spreadsheets remain fundamental tools for data organization, analysis, and reporting across various government branches, including the judiciary and statistical bodies.

**Integration of Excel in Government Planning and Monitoring Frameworks**

The reliance on spreadsheet software extends beyond basic data management to core planning and monitoring functions. The Center for Disaster Preparedness, in its Inclusive Data Management System Guidebook, explicitly outlines a workflow that proceeds "from KoboToolbox installation, to data analysis using Microsoft Excel, to barangay planning and budgeting" [8]. This methodology, designed for barangay-level disaster risk reduction and management planning, positions Excel as the primary analytical tool for transforming raw data into actionable plans. The guidebook integrates the Department of the Interior and Local Government's BDRRMP Template and Quality Assessment Tool, suggesting that spreadsheet-based templates are embedded in official planning frameworks [8].

Academic research on government monitoring systems also reveals the centrality of spreadsheet applications. A 2012 study on the enhanced Regional Project Monitoring and Evaluation System (RPMES) On-line in the Bicol Region documented that while the system provided an interactive website for project information, the underlying processes involved significant manual data handling [7]. The database named "Project Tracking System" was maintained in the local area network of NEDA Region 5, and data of projects were "encoded or transferred, validated, reviewed and approved in the database by the user's and supervisors" [7]. This encoding and transfer process typically involves spreadsheet applications as intermediate tools for data preparation before upload to centralized systems.

**Manual Processes in Local Government Project Monitoring**

At the local government level, project monitoring often combines digital tools with manual inspection and documentation procedures. Naga City's Ordinance No. 2025-061, which institutionalizes quality assurance for government infrastructure projects, establishes a Project Monitoring Committee tasked with "regular inspections" including at least twice during construction and annually during the Defects Liability Period and Warranty Period [14]. The Committee prepares "Inspection and Evaluation Reports with photo/video evidence," and citizens may report defects "via an online platform with photo/video documentation" [14]. While the ordinance provides for online reporting, the core monitoring function relies on human inspectors generating reports, often using basic office software for documentation and analysis.

**Implications for Predictive Analytics**

The widespread reliance on manual processes and spreadsheet-based tools has significant implications for the implementation of predictive analytics systems like MAAGAP. First, it indicates that many government agencies operate at relatively low levels of data management maturity, where basic data organization and cleaning consume substantial personnel time. Second, the prevalence of training programs in Excel and Google Sheets suggests that investments in more advanced analytics tools must be accompanied by significant capacity building. Third, the existence of spreadsheet-based templates embedded in official planning frameworks means that any advanced system must be capable of interfacing with or replacing these familiar tools.

Most importantly, the manual orientation of current processes means that valuable data on project performance, delays, and resource utilization often remains trapped in disconnected spreadsheets, email attachments, and paper reports. This fragmentation prevents the aggregation of historical data needed to train predictive models and limits agencies' ability to identify patterns across project portfolios. MAAGAP's contribution lies not only in its predictive algorithms but in its potential to transform fragmented manual processes into integrated, data-driven workflows that enable anticipatory project management.

**TABLE I COMPARATIVE ANALYSIS OF PHILIPPINE GOVERNMENT PROJECT MANAGEMENT SYSTEMS AND MAAGAP FEATURES**

| System / Framework | Real-Time Tracking & Transparency | Geospatial & Satellite Integration | Predictive Analytics (Delay/Cost Forecasting) | Dynamic Risk Scoring | Prescriptive Resource Optimization | Explainable AI (XAI) |
|---|---|---|---|---|---|---|
| DPWH Transparency Portal | Yes | Yes | No | No | No | No |
| Project DIME (DBM) | Yes | Yes | No | No | No | No |
| PhilGEPS | Yes (Procurement) | No | No | No | No | No |
| COA PPA Disclosure | No (Retrospective) | No | No | No | No | No |
| PhilSA-DPWH Space Data | Yes | Yes | No (Currently retrospective) | No | No | No |
| Manual / Spreadsheet Tools | No (Fragmented) | No | No | No | No | No |
| MAAGAP (Proposed Study) | Yes | Yes | Yes | Yes | Yes | Yes |

### Related Systems or Solutions

#### 2.2 AI-Driven Predictive Systems in Construction and Project Management

While Section 2.1 examined current monitoring systems within the Philippine government, this section reviews existing commercial and specialized platforms that leverage artificial intelligence for project management and construction oversight. These systems represent the technological frontier that MAAGAP seeks to adapt for the Philippine public sector context.

##### 2.2.1 Commercial AI Construction Platforms

The construction and engineering industry has seen the emergence of enterprise-grade AI platforms designed to optimize project delivery through predictive analytics. These systems are typically deployed by large contractors, engineering firms, and infrastructure owners to manage risk across their project portfolios.

**Oracle Construction Intelligence Cloud** represents a comprehensive suite of AI and analytics solutions purpose-built for the engineering and construction industry [48]. Developed by Oracle Corporation, the platform leverages machine learning to deliver predictive intelligence that helps organizations uncover potential future risks, assess their impact, and take proactive actions. The Construction Intelligence Cloud Advisor continuously identifies hidden risks requiring attention, such as schedule delays, poor schedule quality, budget over-runs, resource bottlenecks, and process inefficiencies [49]. Unlike traditional business intelligence tools that generate lag indicators focusing on historical performance, this system provides advanced warnings by predicting what can happen and identifying underlying causes. The AI models have been trained on decades of construction project data, with the company's 2022 acquisition of Newmetrix further enhancing capabilities for safety risk detection from construction site imagery [78]. Newmetrix's AI, trained on ten years of outcome data from contractors including Suffolk Construction, can identify over 100 different safety risks from project documentation [78]. Target users include project owners, general contractors, and program managers, with documented adoption by organizations including Northwell Health, The Boldt Company, Suffolk Construction, and Severn Trent [48].

**ALICE Technologies** offers what the company describes as the world's first generative scheduling platform for construction, launched by a Stanford University spin-off [11]. The platform's core innovation lies in its ability to generate and evaluate millions of scheduling scenarios, enabling project teams to rapidly test various strategies and select the most efficient, cost-effective solution with reduced risk [12]. According to industry analysis from CIMdata, the platform's patented generative scheduling technology enables scenario exploration at speeds 800 times faster than traditional manual approaches [13]. In 2025, ALICE introduced the Insights Agent, a conversational interface that allows users to interrogate schedules using plain language queries [11]. The system supports multilingual interaction and has demonstrated measurable results in deployments: a case study documented by the UK's Get It Right Initiative (GIRI) featured ALICE's collaborative work with Align JV on the HS2 high-speed rail project, highlighting the platform's effectiveness in raising productivity and mitigating risk [30]. Zachry Construction Corporation, a leader in heavy civil construction, has implemented ALICE Core across several key projects, with Vice President of Project Controls Ranjeet Gadhoke noting that the platform enables the team to "evaluate risks and opportunities across our projects in a fraction of the time it previously took" [13].

**Unanet** provides an AI-first Enterprise Resource Planning (ERP) platform specifically designed for government contractors (GovCon) and architecture, engineering, and construction (AEC) firms [72]. The platform's ChampAI introduces multi-agentic capabilities across the Unanet ecosystem, complemented by specialized tools including OpportuneAI for identifying high-value business opportunities and ProposalAI for accelerating proposal generation. For AEC firms specifically, Unanet offers AI-powered Accounts Payable and Accounts Receivable Automation that streamlines billing and payment processes, along with Spend Management for unified expense tracking [72]. Notably, Unanet ERP GovCon has achieved FedRAMP Ready status, an important milestone for government contractors pursuing cybersecurity certifications, demonstrating the platform's alignment with stringent public sector security requirements. The system targets firms needing integrated project management with compliance capabilities for government contracting [72].

##### 2.2.2 AI-Enhanced Project Management Software

Beyond specialized construction platforms, general project management software has increasingly incorporated predictive analytics capabilities to identify project health issues and resource bottlenecks before they materialize.

**Forecast** is an AI-native project and resource management platform that enables professional service teams to efficiently plan, execute, and monitor projects [28]. According to the company's official documentation, the platform's artificial intelligence features, collectively branded as Nova Insights, provide four intelligent capabilities: predicted project end dates, team focus analysis, task performance monitoring, and budget tracking for fixed-price projects [28]. These AI capabilities learn from data captured within the user's account to provide smart suggestions for task assignments and work estimates [28]. The platform's AI for Resourcing analyzes team members' roles, skills, and levels to suggest the most suitable and available resources for projects, while Task Overrun Alerts monitor tasks at risk of exceeding time estimates [28]. The platform serves organizations ranging from 25 to 200+ employees across various industries and has been adopted by companies including Spryker, which reported a 50% reduction in time spent on resource management after implementation [27]. According to the company's official website, Forecast was acquired by Accelo in 2025, combining its AI-driven capacity planning capabilities with comprehensive project management solutions [28].

**Sharktower** delivers AI-powered portfolio and project management with emphasis on predictive analytics and bias-free decision-making [73]. Developed to address the challenge of organizations wasting resources on mismanaged change projects, with nearly 50% of projects failing to meet timescales, Sharktower removes wasteful manual reporting and helps project teams spot problems before they occur [73]. The platform provides portfolio insights visualization and predictive analytics that enable teams to identify and mitigate problems proactively. By underpinning project delivery with continuous data analysis, Sharktower offers transparency and clarity in one unified interface, visualizing where project teams need to focus to increase productivity and the probability of successful outcomes. The platform has been supported by the University of Edinburgh's Bayes Centre AI Accelerator program, indicating institutional validation of its technical approach [73].

##### 2.2.3 AI-Powered Infrastructure Monitoring Systems

Specialized infrastructure monitoring systems have emerged that combine satellite data, sensors, and artificial intelligence to assess risks to physical assets and construction projects.

**SAFER (Seismic Hazard Forecasting & Evaluation for Resilience)**, developed by WAY4WARD with European Space Agency support, provides real-time seismic hazard maps and forecasts by processing satellite Earth Observation data with GNSS-based ground deformation measurements [25]. The system uses proprietary algorithms and AI models to generate hyperlocal seismic risk maps down to 50-100 meter resolution, significantly improving upon traditional methods that rely on historical data and static hazard maps. Target users include government agencies requiring hyperlocal seismic hazard mapping for urban planning, infrastructure operators needing accurate forecasts to strengthen resilience, and large construction firms requiring comprehensive hazard assessments for infrastructure projects [25]. The system delivers outputs through visually enriched reports designed specifically for non-technical decision-makers, enabling proactive disaster preparedness and improved infrastructure resilience. A proof of concept was successfully completed with an early adopter customer in Argentina, demonstrating effective integration of satellite and ground data to produce detailed seismic hazard risk estimations [25].

##### 2.2.4 Government-Focused AI Platforms

Some platforms specifically target the unique requirements of government agencies and public sector project management.

Unanet's government contractor focus represents a significant example of AI platforms tailored to public sector compliance needs. With FedRAMP Ready status and tools designed to support government contractors pursuing Cybersecurity Maturity Model Certification (CMMC), Unanet demonstrates how AI systems can be adapted for the stringent security and compliance requirements of government work [72]. The platform's capabilities for audit tools, role-based controls, and transparent governance of financial and project data align with public sector accountability requirements [72].

Oracle Construction Intelligence Cloud has been adopted by public sector organizations including UK water company Severn Trent, demonstrating applicability to government-owned infrastructure operators [48]. The platform's ability to integrate data across project and portfolio lifecycles, including planning, scheduling, and construction, into comprehensive datasets makes it suitable for agencies managing multiple public infrastructure projects simultaneously [49]. Engineering News-Record reported that Oracle's construction analytics offerings are informed by years of customer data from applications including P6 and Primavera, which the company uses to inform the recommendations made by both its Analytics and Advisor applications [77].

##### 2.2.5 Synthesis and Research Gap

| System / Platform | Predictive Analytics (Delays/Risks) | Prescriptive Resource Optimization | Public Sector Compliance / Frameworks | Explainable AI (XAI) | Support for Fragmented / Semi-Manual Data |
|---|---|---|---|---|---|
| Oracle Construction Intelligence Cloud | Yes | Yes | No (Western Enterprise Focus) | No | No (Requires decades of clean data) |
| ALICE Technologies | Yes | Yes (Generative) | No | No | No (Requires clean pipelines) |
| Unanet ERP GovCon | No (ERP/Financial focus) | Yes | Yes (US-only: FedRAMP/CMMC) | No | No |
| Forecast | Yes (Task overruns) | Yes | No | No | No |
| Sharktower | Yes | Yes | No | No | No |
| SAFER | Yes (Seismic only) | No | No | No | No |
| MAAGAP (Proposed Study) | Yes | Yes | Yes (Philippine RA 9184 & COA) | Yes | Yes (Designed for imperfect data) |

The systems reviewed in this section demonstrate that AI-driven predictive analytics for project management is a mature and commercially validated technology. Enterprise platforms like Oracle Construction Intelligence Cloud and ALICE Technologies have proven capable of predicting delays, identifying risks, and optimizing schedules using machine learning algorithms trained on historical project data, with documented case studies from organizations including Suffolk Construction and Zachry Construction Corporation validating their effectiveness [78, 13]. General project management tools like Forecast and Sharktower have successfully integrated predictive analytics to spot problems before they occur and recommend resource allocations, with Forecast's official documentation detailing specific AI capabilities for task overrun prediction and resource optimization [28]. Specialized infrastructure monitoring systems like SAFER demonstrate the feasibility of combining satellite data with AI for risk assessment, supported by European Space Agency funding and validation [25].

However, several critical gaps emerge when considering the applicability of these systems to the Philippine government context. First, these platforms are designed for private enterprise or Western government agencies with mature data infrastructure, established project management practices, and significant technology budgets. They assume the availability of comprehensive, digitized historical project data, an assumption that does not hold in contexts where manual processes and spreadsheet-based record-keeping remain prevalent, as documented in Section 2.1.6. The training of Newmetrix's AI on ten years of structured outcome data from Suffolk Construction [78] exemplifies the data requirements that Philippine agencies, with fragmented record-keeping, cannot currently meet.

Second, these systems operate within legal and procurement frameworks significantly different from the Philippines. They are not designed to account for the specific requirements of Republic Act No. 9184 (Government Procurement Reform Act), Commission on Audit reporting requirements, or the transparency mandates embedded in Philippine open government directives. A predictive model that does not understand these constraints could generate recommendations that are technically optimal but legally non-compliant.

Third, the explainability requirements for public sector AI differ substantially from private sector contexts. While commercial platforms provide explanations sufficient for corporate decision-makers, Philippine government agencies face constitutional transparency obligations and public accountability demands that require more rigorous interpretability. Citizens and oversight bodies must be able to understand why a system flagged a particular project as high-risk or recommended specific resource reallocations.

Fourth, none of the reviewed systems address the specific challenge of integrating predictive analytics with the fragmented, semi-manual data environments characteristic of Philippine local government units. They assume clean, integrated data pipelines rather than the reality of disconnected spreadsheets, manual encoding processes, and paper-based documentation documented in Section 2.1.

MAAGAP addresses these gaps by adapting established machine learning methodologies, including classification algorithms demonstrated effective in academic research, to the specific constraints of the Philippine public sector. Rather than requiring comprehensive digitized historical data, MAAGAP's methodology incorporates strategies for working with limited, imperfect data through appropriate preprocessing and validation techniques. Rather than assuming Western procurement frameworks, the system's optimization algorithms incorporate Philippine-specific constraints including budget rules, COA reporting requirements, and transparency mandates. Rather than providing explanations sufficient only for internal users, MAAGAP's interpretability module is designed to support public accountability through clear presentation of feature importance and model reasoning.

The systems reviewed in this section validate the technical feasibility of MAAGAP's core objectives. Oracle and ALICE demonstrate that machine learning can effectively predict construction delays and optimize schedules, with independent case studies from organizations like Zachry Construction Corporation providing validation [13]. Forecast's official documentation confirms that AI capabilities for resource optimization and task overrun prediction are mature enough for enterprise deployment [28]. SAFER illustrates the value of integrating external data sources for infrastructure risk assessment, with the European Space Agency backing ensuring technical credibility [25]. MAAGAP's contribution lies not in inventing new algorithms, but in integrating these proven capabilities into a unified platform designed specifically for the legal, institutional, and data environment of Philippine government project management, an adaptation that existing commercial systems do not provide.

### Related Studies

While Sections 2.1 and 2.2 examined implemented government systems and commercial AI platforms, this section reviews empirical academic research on the application of machine learning to project management, risk prediction, and resource allocation. These studies provide the methodological foundations and empirical evidence that inform MAAGAP's design.

#### 2.3.1 Ensemble Machine Learning Methods for Construction Delay Prediction

Ensemble learning methods, which combine multiple base models to improve predictive performance, have emerged as a dominant approach in construction delay prediction research due to their ability to capture complex, nonlinear relationships in project data.

**Sahu et al.** conducted a comprehensive investigation of supervised machine learning algorithms specifically for construction delay prediction, with direct relevance to MAAGAP's predictive analytics module [63]. Published in the Journal of Mechanics of Continua and Mathematical Sciences (Scopus-indexed), this study evaluated seven algorithms: Gaussian Naïve Bayes, Adaboost, Logistic Regression, Gradient Boosting (GB), Random Forest (RF), Decision Tree (DT), and Extreme Gradient Boosting (XGBoost). The models were assessed using accuracy, precision, recall, and F1 score metrics to validate their real-world applicability. The results demonstrated that ensemble learning techniques---specifically Random Forest and XGBoost---significantly outperformed simpler models, exhibiting superior accuracy and predictive capacity. The study attributed this performance to the ensemble methods' ability to capture convoluted relationships inherent in construction data, making them particularly suitable for project risk management applications. In contrast, simpler models like Adaboost and Gaussian Naïve Bayes, while more interpretable, demonstrated lesser predictive accuracy and were consequently less suitable for construction delay forecasting. The authors emphasized that ensemble-based delay prediction models have the potential to reduce project uncertainties, control delays, schedule resources efficiently, and ultimately improve infrastructure project cost efficiency and timely completion. However, they identified critical barriers to widespread adoption including data quality challenges, model interpretability requirements, and integration with real-time project management systems [63].

**Gondia et al.** developed machine learning algorithms specifically for construction projects delay risk prediction in a study published in the Journal of Construction Engineering and Management (ASCE), one of the most authoritative journals in the field [34]. This Scopus-indexed study employed a dataset of historical construction projects in Egypt to develop and validate multiple machine learning models for delay risk prediction. The research compared various algorithms including decision trees, support vector machines, and ensemble methods, finding that ensemble approaches demonstrated superior performance in identifying projects at risk of delay before problems manifested. The study's methodological contribution lay in its systematic approach to feature engineering, identifying key project characteristics that most strongly predicted delay outcomes. The findings validated the applicability of machine learning to construction delay prediction in Middle Eastern contexts, though the authors noted that model performance remained sensitive to data quality and the specific characteristics of regional construction practices [34].

**Sanni-Anibire, Zin, and Olatunji** developed a machine learning-based framework for construction delay mitigation, published in the Journal of Information Technology in Construction (ITcon), a leading open-access journal in construction informatics [64]. This study focused specifically on tall building projects, which present unique delay challenges due to their complexity and height. The research employed a dataset of tall building projects and evaluated multiple machine learning algorithms including decision trees, random forests, and neural networks for delay risk assessment. The random forest model demonstrated particularly strong performance, achieving high accuracy in classifying projects into risk categories. The authors emphasized that the framework's practical value lay in its ability to provide early warnings to project managers, enabling proactive intervention before delays escalated. The study explicitly addressed the interpretability challenge by analyzing feature importance rankings from the random forest model, identifying the most influential factors contributing to delay predictions [64].

In a related study, **Sanni-Anibire, Zin, and Olatunji** further refined their approach, developing a machine learning model specifically for delay risk assessment in tall building projects [65]. Published in the International Journal of Construction Management (Taylor & Francis), this research employed a dataset of tall building projects and compared multiple algorithms, finding that ensemble methods again outperformed single classifiers. The study's contribution lay in its systematic feature selection process, identifying critical delay factors specific to tall building construction, and its demonstration that machine learning models could achieve practical accuracy levels suitable for decision support. The authors noted that while their models were developed and validated on tall building projects, the methodological approach could be adapted to other construction types [65].

**Yaseen et al.** developed a hybrid artificial intelligence model for predicting risk delay in construction projects, published in Sustainability (MDPI) [76]. This Scopus-indexed study employed a hybrid approach combining multiple AI techniques to enhance prediction accuracy. The research addressed the limitation that single models often fail to capture the full complexity of construction delay factors, proposing instead an integrated methodology that leveraged the strengths of different algorithms. The hybrid model demonstrated improved performance over individual classifiers, validating the ensemble approach that MAAGAP adopts through its combination of Random Forest, Gradient Boosting, and LSTM networks. The study emphasized that prediction accuracy alone was insufficient; models must also provide actionable insights for project managers [76].

**Egwim et al.** conducted a systematic review of critical drivers for delay risk prediction, working toward a conceptual framework for BIM-based construction projects [20]. Published in Frontiers in Engineering and Built Environment (Emerald), this review synthesized findings from multiple studies to identify the most significant predictors of construction delays. The authors found that ensemble machine learning methods consistently outperformed traditional statistical approaches across diverse contexts, validating the methodological choices in studies like Sahu et al. [63] and Gondia et al. [34]. The review also highlighted the emerging integration of machine learning with Building Information Modeling (BIM) as a promising direction for future research, enabling richer feature sets and more accurate predictions through access to detailed project data [20].

In a related empirical study, **Egwim et al.** applied artificial intelligence for predicting construction project delays, published in Machine Learning with Applications (Elsevier) [19]. This research employed multiple machine learning algorithms on construction project datasets, finding that ensemble methods including Random Forest and Gradient Boosting achieved superior performance compared to single classifiers. The study's practical contribution lay in demonstrating that machine learning models could achieve accuracy levels sufficient for real-world deployment, provided that training data adequately represented the target project contexts. The authors emphasized the importance of local data for model training, noting that models trained on projects from one geographic or regulatory context may not generalize to others [19].

#### 2.3.2 Deep Learning and Sequential Models for Project Data

While ensemble methods excel at capturing complex nonlinear relationships, deep learning approaches---particularly Long Short-Term Memory (LSTM) networks---offer advantages for modeling temporal dependencies in sequential project data.

**Alsulamy** conducted a comparative analysis of deep learning algorithms for predicting construction project delays in Saudi Arabia, examining various neural network architectures and their effectiveness in forecasting timeline deviations [2]. Published in Applied Soft Computing (Elsevier), this Scopus-indexed study evaluated multiple deep learning approaches against traditional machine learning methods, finding that properly configured deep neural networks significantly outperformed conventional techniques when sufficient historical data were available. The research emphasized that model selection must be tailored to the specific characteristics of construction project data, including temporal dependencies and nonlinear relationships among delay factors. The study contributes to the growing body of evidence supporting deep learning applications in construction delay prediction while acknowledging the critical role of data quality and availability [2].

**Mostofi, Tokdemir, and Toğan** developed an innovative approach to construction delay prediction using a relationship-aware multihead graph attention network (GAT), published in the Journal of Management in Engineering (ASCE) [47]. This Scopus-indexed study addressed a fundamental limitation of existing machine learning delay prediction models: their inability to process dependencies among construction progress records. By leveraging attention mechanisms, the GAT model emphasized differential node significance in networks and demonstrated the capability to learn input configurations. The dataset was configured into six networks linking records based on contractual alignment and spatial proximity dependency criteria. Under contractual alignments, predictions for electrical and concrete tasks achieved 65% and 76% accuracy respectively, outperforming spatial-based predictions. However, multihead GAT with spatial networks delivered 77% accuracy for insulation tasks, surpassing the 67% achieved by contractual networks. These results underscored model sensitivity to task dependencies and demonstrated applicability across a range of decision-making contexts. The authors emphasized that by recognizing dependencies and shared aspects among construction records, their GAT model better reflects human understanding of construction progress reports, shifting the focus from mere predictive accuracy to representative modeling of construction delay [47].

**Taha, Ibrahim, and Soliman** developed a risk-indexed artificial neural network for predicting duration and cost in Egyptian irrigation canal lining projects, where uncertainties frequently lead to delays and budget overruns [68]. Published in Scientific Reports (Nature Portfolio), this Scopus-indexed study first reduced ninety-three factors to twenty using Analytic Hierarchy Process-Relative Importance Index (AHP-RII) with high reliability (Cronbach's α = 0.954). A multi-layer perceptron with 128-64-32 architecture employing ReLU activation and Adam optimization was trained on 5,000 simulated scenarios and validated on eight real-world projects using leave-one-project-out cross-validation. The model achieved R² = 0.92 for training and 0.82 for testing, with average prediction errors of 0.87 months for duration and EGP 102,500 for cost. The research introduced an integrated ANN-based framework combining expert-driven risk assessment with machine learning, deployed as a Python-based desktop application for practical use by engineers and planners during early project stages. The authors noted that while previous studies had applied machine learning techniques such as regression models and Monte Carlo simulations to address uncertainty in project estimation, few had operationalized these models into practical tools incorporating risk-based adjustments for both time and cost [68].

#### 2.3.3 Comparative Analysis: Ensemble Methods vs. Deep Learning

The studies reviewed reveal important patterns regarding the relative performance of different machine learning approaches for construction delay prediction.

**Ensemble Methods Dominance:** Multiple studies converge on the finding that ensemble methods---particularly Random Forest and XGBoost---consistently outperform simpler algorithms for construction delay prediction. Sahu et al. directly compared multiple algorithms including ensemble methods (Random Forest, XGBoost, Gradient Boosting) against simpler classifiers, finding that ensemble approaches consistently achieved superior accuracy [63]. This finding aligns with Gondia et al., who similarly reported that ensemble methods outperformed single classifiers in delay risk prediction [34]. The superiority of ensemble methods can be attributed to their ability to reduce variance through combining multiple models, capture complex nonlinear relationships, and handle the heterogeneous feature types characteristic of construction project data.

**Deep Learning for Temporal and Relational Data:** For projects where temporal dependencies or complex task relationships are critical, deep learning approaches offer complementary advantages. While not directly compared in a single study, the evidence across studies suggests that deep learning may be preferable when temporal dependencies are critical and sufficient training data are available [2, 47]. The graph attention network approach developed by Mostofi et al. demonstrates particular promise for modeling task dependencies that traditional models ignore [47]. However, deep learning models typically require larger training datasets than ensemble methods---a significant consideration in data-scarce environments like Philippine local government units.

The choice between ensemble and deep learning approaches thus involves trade-offs between interpretability, data requirements, and the specific nature of prediction tasks. Random Forest models offer inherent interpretability through feature importance rankings, a critical advantage for public sector applications requiring explainability [63, 64]. LSTM networks, while potentially more accurate for sequential data, operate as "black boxes" unless explicitly augmented with explainability techniques---an approach MAAGAP adopts through its commitment to explainable AI principles.

#### 2.3.4 AI-Driven Decision Support and Resource Allocation

**Almalki** investigated AI-driven decision support systems for enhancing risk mitigation and resource allocation in Agile software project management [1]. Published in Systems (MDPI), this Scopus-indexed study introduced an AI-based decision support system merging optimization frameworks with predictive analytics to enhance operational decision efficiency. The machine learning solution anchored data evaluation using AI models that simultaneously predicted risks and strengthened decision power for resource scheduling. Testing relied on project records and recent operational data for model validation and training. The framework demonstrated 94% accuracy in risk identification, enhanced workload management by 25%, and led to an 18% improvement in sprint completion rates, outperforming traditional Agile applications. The findings confirmed that AI-driven decision support systems are crucial in enhancing project management by enabling proactive risk mitigation and optimized resource allocation. However, the study's focus on software development projects rather than physical infrastructure limits direct applicability to the construction domain, though the methodological approach to integrating prediction and optimization remains relevant [1].

A comparative analysis published in Future Internet (MDPI) evaluated seven machine learning algorithms for dynamic workload management in the public sector [18]. This Scopus-indexed study addressed the critical challenge of efficient human resource management in public sector environments where traditional systems struggle to adapt to fluctuating workloads. Using a dataset encompassing public and private sector experience, educational history, and age, the researchers evaluated Linear Regression, Artificial Neural Networks (ANNs), Adaptive Neuro-Fuzzy Inference System (ANFIS), Support Vector Machine (SVM), Gradient Boosting Machine (GBM), Bagged Decision Trees, and XGBoost. Performance was assessed through ten evaluation metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). The results demonstrated ANFIS as the superior model, consistently outperforming other algorithms across all metrics by synergizing fuzzy logic's capacity to model uncertainty with neural networks' adaptive learning, enabling precise capability predictions in dynamic environments. This study is particularly relevant for MAAGAP's optimization module, as it validates the effectiveness of adaptive algorithms for resource allocation under the uncertain conditions characteristic of public sector operations [18].

**Choo et al.** developed a learning-augmented approach to resource allocation under budget uncertainty for Ethiopia's Ministry of Health as part of nationwide efforts aligned with the United Nations' Sustainable Development Goal 3 on Universal Health Coverage [10]. This research, available as an arXiv preprint, produced the Health Access Resource Planner (HARP), a decision-support optimization framework for sequential facility planning that aimed to maximize population coverage under budget uncertainty while satisfying region-specific proportionality targets. Two algorithms were proposed: a learning-augmented approach that improved upon expert recommendations at any single step, and a greedy algorithm for multi-step planning, both with strong worst-case approximation guarantees. In collaboration with the Ethiopian Public Health Institute and Ministry of Health, the empirical efficacy of the method was demonstrated in three regions across various planning scenarios. As a preprint, this work has not yet undergone formal peer review, but its grounding in a real-world government partnership and explicit engagement with public sector constraints (region-specific targets, budget uncertainty) offers valuable methodological insights for government-focused optimization systems [10].

#### 2.3.5 Machine Learning for Government Project Performance Analysis

**Strang** explored IT project performance prediction from government big data using supervised machine learning from a managerial perspective [67]. Published in the International Journal of Business Performance Management (Inderscience), this peer-reviewed study noted that longitudinal studies across industries indicate approximately half of all information technology projects are not considered successful, with significant losses due to cybersecurity breaches. Given the high volume of project performance metrics available, the study analyzed numerous IT project features from government big data to determine if and how supervised machine learning could explain performance. Ten features were identified to predict IT project performance success through classification and regression ML techniques, with effect sizes near 54%. Critically, all ML processes were explained and interpreted in business language so that decision-makers and researchers could understand the results, generalize the implications, and apply ML in their practice areas. The study's emphasis on interpretability and managerial accessibility aligns with MAAGAP's commitment to explainable AI for public sector stakeholders [67].

A study on leveraging artificial intelligence and data analytics for decision-making in large IT projects characterized by persistent instability and escalating budgetary and schedule risks was conducted at PwC CEE IT practice and made available through the CORE repository [38]. The research described mechanisms for deploying predictive models and generative AI tools across the IT project life cycle, empirically validating claimed effects. The novelty lay in combining predictive machine learning, Monte Carlo simulations, AI scoring, and generative planning in a single decision-making loop, with detailed documentation of implementation and adaptation experiences in a real PMO process. While this work is available as a technical report rather than a peer-reviewed journal article, its detailed documentation of practical implementation in a professional services context provides valuable insights for deploying integrated AI systems in project management environments. The study is particularly relevant for project leaders, PMO analysts, and developers of decision-support systems [38].

#### 2.3.6 Comparative Analysis and Synthesis

Across these studies, several methodological patterns emerge regarding the relative performance of different machine learning approaches for construction delay prediction and project management.

**Ensemble Methods Dominance:** Multiple studies converge on the finding that ensemble methods---particularly Random Forest and XGBoost---consistently outperform simpler algorithms for construction delay prediction. Sahu et al. directly demonstrated this through systematic comparison of seven algorithms, finding ensemble methods superior across multiple performance metrics [63]. Gondia et al. [34] and Sanni-Anibire et al. [64, 65] corroborate these findings in different construction contexts. This convergence across studies, journals, and geographic contexts provides strong evidence that ensemble approaches should form the foundation of any construction delay prediction system.

**Deep Learning for Temporal and Relational Data:** For projects where temporal dependencies or complex task relationships are critical, deep learning approaches offer complementary advantages. Mostofi et al.'s graph attention network explicitly models task dependencies, achieving 65-77% accuracy depending on task type and network configuration [47]. Taha et al.'s multi-layer perceptron achieved R² = 0.92 for irrigation project duration prediction when combined with expert-driven risk assessment [68]. However, deep learning approaches generally require larger datasets than ensemble methods---a significant consideration for deployment in data-scarce environments.

**Integration of Prediction and Optimization:** Studies that integrate predictive and prescriptive capabilities remain rare, with Almalki's framework for Agile software development being a notable exception [1]. This framework demonstrated that integrated systems can achieve 94% risk identification accuracy while simultaneously improving resource allocation---validating the integrated approach MAAGAP proposes, though in a different domain.

**Public Sector Specificity:** Studies explicitly addressing government or public sector contexts remain limited. The public sector workload management study [18] and Strang's government IT project analysis [67] represent important exceptions, validating that machine learning can be effectively applied to public sector data. The Ethiopian health resource study, while a preprint, demonstrates the value of engaging directly with government partners and incorporating region-specific constraints into optimization frameworks [10].

**Interpretability as Critical Requirement:** Across multiple studies, interpretability emerges as a critical concern for practical deployment. Sahu et al. explicitly note model interpretability as a barrier to widespread adoption [63]. Strang emphasizes translating ML processes into business language accessible to decision-makers [38]. Sanni-Anibire et al. leverage feature importance rankings from random forest models to provide actionable insights [64, 65]. This consistent emphasis validates MAAGAP's commitment to explainable AI principles.

#### 2.3.7 Research Gap

The empirical literature validates the technical feasibility of MAAGAP's core objectives. Studies confirm that ensemble machine learning methods---specifically Random Forest and XGBoost---can effectively predict construction project delays [63, 34, 64, 65, 19]. Deep learning approaches, including LSTM networks and graph attention mechanisms, offer complementary capabilities for modeling temporal dependencies and task relationships [2, 47, 68]. Decision support frameworks integrating predictive and prescriptive capabilities have demonstrated significant improvements in risk identification accuracy and resource allocation in related domains [1, 18]. Government-specific studies validate that machine learning can be effectively applied to public sector data when interpretability and managerial accessibility are prioritized [67, 38].

However, four critical gaps emerge that MAAGAP directly addresses:

**First**, no existing study integrates ensemble-based delay prediction, dynamic risk scoring, AND resource optimization into a unified framework specifically designed for government infrastructure project portfolios. The literature treats prediction [63, 34, 64], risk assessment [65, 76], and optimization [18, 10] as separate problems solved by separate systems. The sole integrated example focuses on software rather than physical infrastructure [1].

**Second**, explainability requirements for public sector accountability---constitutional transparency mandates, citizen oversight, auditability---remain underexplored. While studies acknowledge interpretability as important [67, 63, 64], none embed explainable AI principles as fundamental design requirements from inception, as MAAGAP does through its commitment to providing clear reasoning behind predictions and recommendations.

**Third**, existing research assumes institutional and legal contexts that differ significantly from the Philippines. No studies address the specific constraints of Republic Act No. 9184 (Government Procurement Reform Act), Commission on Audit reporting requirements, or the transparency mandates embedded in Philippine open government directives. The Ethiopian health study [10] demonstrates the value of engaging with government-specific constraints, but its focus on health facility planning differs substantially from infrastructure project management.

**Fourth**, the data environment of Philippine local government units---characterized by fragmented spreadsheets, manual processes, and limited historical digitization---differs fundamentally from the clean, integrated datasets assumed in most reviewed studies. While the Egyptian irrigation study's use of simulated data offers one approach to data-scarce environments [68], the specific challenges of integrating predictive analytics with semi-manual government workflows remain unaddressed.

MAAGAP addresses these gaps by: (1) integrating ensemble-based delay prediction, risk scoring, and optimization into a unified platform for infrastructure projects; (2) embedding explainable AI principles from inception to support public accountability; (3) adapting algorithms to Philippine legal and procurement frameworks; and (4) developing methodologies appropriate for semi-manual data environments. The empirical literature confirms that each component is technically achievable; MAAGAP's contribution lies in their integrated adaptation to the specific context of Philippine government project management.

**Table III EMPIRICAL EVALUATION RESULTS OF MACHINE LEARNING ALGORITHMS IN RELATED STUDIES**

| Study / Authors | Feature / Attribute | Details / Evaluation Results |
|---|---|---|
| 1. Sahu et al. | Domain Focus | Construction delay prediction. |
| | Algorithms Evaluated | Gaussian Naïve Bayes, Adaboost, Logistic Regression, Gradient Boosting (GB), Random Forest (RF), Decision Tree (DT), and Extreme Gradient Boosting (XGBoost). |
| | Evaluation Metrics | Accuracy, precision, recall, and F1 score. |
| | Key Results | Ensemble learning techniques (specifically RF and XGBoost) significantly outperformed simpler models. Simpler models like Adaboost and Gaussian Naïve Bayes demonstrated lesser predictive accuracy. |
| 2. Gondia et al. | Domain Focus | Construction projects delay risk prediction in Egypt. |
| | Algorithms Evaluated | Decision trees, support vector machines, and ensemble methods. |
| | Key Results | Ensemble approaches demonstrated superior performance in identifying projects at risk of delay before problems manifested. |
| 3. Sanni-Anibire et al. | Domain Focus | Delay risk assessment and mitigation in tall building projects. |
| | Algorithms Evaluated | Decision trees, random forests, neural networks, and various single classifiers. |
| | Key Results | The random forest model achieved high accuracy in classifying projects into risk categories. Ensemble methods consistently outperformed single classifiers. |
| 4. Egwim et al. | Domain Focus | Predicting construction project delays. |
| | Algorithms Evaluated | Random Forest, Gradient Boosting, single classifiers, and traditional statistical approaches. |
| | Key Results | Ensemble machine learning methods (RF and GB) achieved superior performance and consistently outperformed both traditional statistical approaches and single classifiers. |
| 5. Alsulamy | Domain Focus | Predicting construction project delays in Saudi Arabia. |
| | Algorithms Evaluated | Deep learning algorithms (deep neural networks) vs. traditional machine learning methods. |
| | Key Results | Properly configured deep neural networks significantly outperformed conventional techniques, provided sufficient historical data were available. |
| 6. Mostofi et al. | Domain Focus | Construction delay prediction based on task dependencies. |
| | Algorithms Evaluated | Relationship-aware multihead graph attention network (GAT). |
| | Key Results | Contractual alignments networks achieved 65% accuracy for electrical tasks and 76% for concrete tasks. Spatial networks delivered 77% accuracy for insulation tasks, surpassing the 67% from contractual networks. |
| 7. Taha et al. | Domain Focus | Predicting duration and cost in Egyptian irrigation canal lining projects. |
| | Algorithms Evaluated | Multi-layer perceptron (128-64-32 architecture, ReLU activation, Adam optimization) combined with AHP-RII. |
| | Key Results | Achieved R² = 0.92 for training and 0.82 for testing. Average prediction errors were 0.87 months for duration and EGP 102,500 for cost. |
| 8. Almalki | Domain Focus | Enhancing risk mitigation and resource allocation in Agile software projects. |
| | Algorithms Evaluated | AI-based decision support system merging optimization with predictive analytics. |
| | Key Results | Framework demonstrated 94% accuracy in risk identification, enhanced workload management by 25%, and led to an 18% improvement in sprint completion rates. |
| 9. MDPI Future Internet Study | Domain Focus | Dynamic workload and human resource management in the public sector. |
| | Algorithms Evaluated | Linear Regression, ANNs, ANFIS, SVM, GBM, Bagged Decision Trees, and XGBoost. |
| | Evaluation Metrics | Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). |
| | Key Results | Adaptive Neuro-Fuzzy Inference System (ANFIS) consistently outperformed all other algorithms across all metrics. |
| 10. Strang | Domain Focus | IT project performance prediction from government big data. |
| | Algorithms Evaluated | Classification and regression ML techniques. |
| | Key Results | Identified ten features to predict project performance success, achieving effect sizes near 54%. |

---

# CHAPTER 3 RESEARCH DESIGN AND METHODOLOGY

## Description of the Proposed Study

This study employs the Design Science Research (DSR) methodology to develop MAAGAP (Machine Analytics for Allocation, Governance, and Assessment of Projects), a machine learning framework for predictive risk assessment and optimized resource allocation in government project management. DSR is a research approach that creates and evaluates innovative artifacts to solve practical problems in information systems [36]. Unlike traditional research that only explains phenomena, DSR focuses on building practical solutions---such as software systems, algorithms, and frameworks---that address real-world challenges [62].

The research follows the DSR process model developed by Peffers et al. [51], which includes six activities: (1) problem identification and motivation, (2) definition of objectives for a solution, (3) design and development, (4) demonstration, (5) evaluation, and (6) communication. This cyclical approach allows continuous improvement of the predictive models and optimization algorithms based on testing results and feedback.

The study addresses persistent challenges in Philippine government infrastructure projects, including schedule delays, budget overruns, and inefficient resource allocation documented in the Commission on Audit reports. MAAGAP integrates three core components: (1) a Predictive Analytics Module using ensemble machine learning and LSTM networks to forecast delays and cost overruns, (2) a Risk Scoring Module that classifies projects into actionable risk categories with automated alerts, and (3) an Optimization Module using linear programming to generate resource reallocation recommendations under budget, manpower, and equipment constraints.

## Methods and Proposed Enhancements

### Predictive Modeling for Project Risk Assessment

This section describes the methodology for achieving the first two objectives: developing ensemble machine learning models to forecast government infrastructure project delays and cost overruns, and designing a dynamic risk scoring engine to classify projects into low, medium, high, and critical categories.

#### Data Collection and Preparation

The study utilizes historical project data from government infrastructure databases, including project timelines, budget allocations, contractor performance records, and geographical metadata. Data preprocessing follows established machine learning protocols:

1. **Data Cleaning:** Missing values are handled through multiple imputation. Outliers in project duration and cost figures are identified using the Interquartile Range (IQR) method and verified for data entry accuracy.

2. **Feature Engineering:** Categorical variables, including project type, funding source, and geographical region, are encoded using one-hot encoding. Temporal features such as project month, seasonality indicators, and days elapsed since project initiation are extracted from timeline data. External contextual factors---weather patterns from PAGASA, economic indicators from the Philippine Statistics Authority, and geographic variables---are integrated to enhance prediction accuracy.

3. **Feature Selection:** Correlation analysis identifies the most informative predictors while reducing dimensionality. Principal Component Analysis (PCA) is applied when multicollinearity is detected among related features.

4. **Data Normalization:** Min-max normalization scales numerical features to the [0,1] range, ensuring equitable contribution across variables with different measurement scales.

#### Model Development: Ensemble Methods

For predicting delays and cost overruns, this study implements ensemble machine learning methods, specifically Random Forest and Gradient Boosting classifiers via scikit-learn, selected for their robust performance, interpretability, and accessibility. Recent research demonstrates that ensemble methods significantly outperform individual models for construction delay prediction [63].

**Random Forest:** This bagging ensemble constructs multiple decision trees using bootstrap aggregating, where each tree is trained on a random subset of the training data with replacement. Final predictions are determined by majority voting for classification tasks. The algorithm's inherent feature importance measures facilitate model interpretability, while the ensemble structure reduces overfitting through variance reduction [63].

**Gradient Boosting (XGBoost):** XGBoost builds trees sequentially, with each new tree correcting errors from previous ones. It includes regularization to prevent overfitting [9]:

$$L^{(t)} = \sum_{i = 1}^{n}{l\left( y_{i},{\widehat{y_{i}}}^{(t - 1)} + f_{t}\left( x_{i} \right) \right)} + \Omega\left( f_{t} \right)$$

Where $l$ represents the loss function, $y_{i}$ the true label, $\widehat{y}_{i}^{(t - 1)}$ the prediction from previous trees, and $\Omega(f_{t})$ the regularization term controlling model complexity.

#### Model Development: LSTM for Temporal Dependencies

To capture temporal dependencies in sequential project data, this study implements Long Short-Term Memory (LSTM) networks using TensorFlow/Keras. LSTM is a type of recurrent neural network that learns long-term dependencies through gating mechanisms---input, forget, and output gates---that regulate information flow, addressing the vanishing gradient problem that constrains standard RNNs.

The LSTM architecture processes time-series project data where each time step represents a project monitoring period. Input features include cumulative expenditure, percentage completion, resource utilization rates, and identified issues at each reporting interval. The network learns to predict future project states based on historical trajectories, enabling early warning of deviation from planned schedules and budgets.

*Figure 1: MAAGAP LSTM Architecture*

#### Model Evaluation

Predictive model performance is evaluated using standard classification and regression metrics:

- **Accuracy:** Proportion of correct predictions among total predictions

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Where TP = true positives (correctly identified at risk projects), TN = true negatives (correctly identified low-risk projects), FP = false positives (low-risk projects incorrectly flagged), and FN = false negatives (at-risk projects missed).

- **Precision:** Ratio of true positive predictions to total positive predictions, measuring the reliability of positive risk classifications

$$\text{Precision} = \frac{TP}{TP + FP}$$

High precision means when MAAGAP flags a project as "at-risk," it is likely correct---minimizing unnecessary alarms that could desensitize users.

- **Recall (Sensitivity):** Ratio of true positive predictions to actual positive cases, measuring ability to detect at-risk projects

$$\text{Recall} = \frac{TP}{TP + FN}$$

High recall ensures MAAGAP catches most genuinely at-risk projects---critical for government accountability where missed delays have public impact.

- **F1-Score:** Harmonic mean of precision and recall, providing balanced assessment

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The F1-score balances the trade-off between precision and recall, useful when you need both reliable alerts AND comprehensive risk detection.

- **Area Under the ROC Curve (AUC-ROC):** Model's ability to distinguish between risk levels across various threshold settings

$$\text{AUC-ROC} = \int_{0}^{1}{\text{TPR}(t)}\, d\left\lbrack \text{FPR}(t) \right\rbrack$$

where TPR = true positive rate (recall) and FPR = false positive rate. AUC-ROC ranges from 0.5 (random guessing) to 1.0 (perfect classification), with values above 0.75 indicating good discriminative ability for MAAGAP's risk categories.

- **Mean Absolute Error (MAE):** Error magnitude for regression predictions

$$\text{MAE} = \frac{1}{n}\sum_{i = 1}^{n}\left| y_{i} - \widehat{y_{i}} \right|$$

Where $y_{i}$ = actual delay duration or cost overrun, and $\widehat{y_{i}}$ = predicted value. MAE measures average prediction error in original units (e.g., days delayed or pesos over budget), making it interpretable for government stakeholders.

#### Risk Classification Framework

The Risk Scoring Module translates continuous probability predictions into discrete risk categories using threshold-based classification. Projects are classified into four actionable tiers: Low (0--0.3), Medium (0.3--0.7), High (0.7--0.9), and Critical (0.9--1.0). Risk scores are computed as weighted combinations of delay probability, cost overrun probability, and project strategic importance.

Logic consistency testing verifies that projects are assigned to correct risk tiers based on defined probability thresholds. The system automatically triggers alerts and notifications when projects transition between risk categories or when predicted risk levels exceed organizational tolerances, enabling timely intervention by project managers and oversight agencies.

### Optimization for Resource Allocation

This section addresses the third objective: creating optimization algorithms using linear programming to generate resource reallocation recommendations with at least 15% improvement in allocation efficiency compared to current manual approaches.

#### Problem Formulation

Given a portfolio of $n$ projects requiring allocation of $m$ resource types (financial, personnel, equipment), the optimization seeks to determine an allocation matrix $X = \left\lbrack x_{ij} \right\rbrack$ where $x_{ij}$ represents the quantity of resource $i$ allocated to project $j$. The problem is formulated as a linear programming optimization problem, providing straightforward implementation while maintaining mathematical rigor.

The objective function maximizes portfolio success probability while minimizing total cost:

$$\text{Maximize } Z = \sum_{j = 1}^{n}w_{j} \cdot P\left( \text{success}_{j} \right) - \lambda\sum_{i = 1}^{m}{\sum_{j = 1}^{n}{c_{ij}x_{ij}}}$$

Subject to:

- Budget constraints: $\sum_{j = 1}^{n}{c_{ij}x_{ij}} \leq B_{i}\quad\text{for each resource type }i$
- Resource availability: $x_{ij} \leq A_{i}\quad\text{for all }i,j$
- Project requirements: $\sum_{i = 1}^{m}x_{ij} \geq R_{j}^{\min}\quad\text{for each project }j$
- Non-negativity: $x_{ij} \geq 0$

Where $w_{j}$ represent project priority weights, $P(success)$ the predicted success probability from the risk scoring module, $c_{ij}$ unit costs, and $\lambda$ the cost-priority tradeoff parameter.

#### NSGA-II Algorithm

The Non-dominated Sorting Genetic Algorithm II (NSGA-II) solves this multi-objective problem by finding Pareto-optimal solutions. Recent research demonstrates NSGA-II's effectiveness for resource allocation in complex networks. A 2025 study applied NSGA-II for multi-objective disaster recovery scheduling in virtual cloud platforms, achieving optimal load balance degrees 21.1% and 19.0% lower than MOEA/D and NSGA-III respectively in small-scale failure scenarios, and 35.7% and 25.0% lower in large-scale disaster scenarios [41].

NSGA-II uses:

1. **Non-dominated sorting:** Classifies solutions into Pareto fronts based on dominance
2. **Crowding distance:** Maintains solution diversity
3. **Genetic operators:** Selection, crossover (PC=0.80), and mutation (Pm=0.05) generate new solutions

Recent studies have applied NSGA-II to various resource allocation problems including multi-UAV maritime search and rescue [43], post-disaster relief distribution [61], and industrial productivity optimization [39]. Enhanced NSGA-II algorithms with simulated annealing-based strategies have shown improved performance for real-world multi-objective optimization problems [40].

#### Optimization Evaluation

Resource allocation performance is evaluated through:

- **Efficiency improvement:** Comparison of optimized allocation against baseline manual approaches, targeting 15% improvement in resource utilization efficiency
- **Feasibility validation:** Expert review confirming recommendations respect real-world constraints
- **Simulation testing:** Monte Carlo simulation assessing robustness under uncertainty

### Model Interpretability and Transparency

To ensure transparent predictions (fourth objective), this study incorporates SHAP (SHapley Additive exPlanations) values for model interpretation. SHAP provides unified, mathematically principled feature attribution based on cooperative game theory, satisfying properties of local accuracy and consistency [42].

For each prediction, SHAP values $\phi_{i}$ quantify feature $i$'s contribution to the prediction difference from baseline:

$$\phi_{i} = \sum_{S \subseteq F \smallsetminus \text{\{i\}}}^{}\frac{|S|!\left( |F| - |S| - 1 \right)!}{|F|!}\left\lbrack f\left( S \cup \text{\{i\}} \right) - f(S) \right\rbrack$$

Where $F$ represents the set of all features, $S$ subsets of other features, and $f(S)$ the model prediction using features in $S$.

Recent research confirms SHAP's effectiveness for explaining machine learning models in clinical decision support systems. A 2025 systematic review found SHAP and LIME are widely used across medical domains including cardiology, oncology, and endocrinology for explaining model predictions, with SHAP providing globally stable attributions [31]. SHAP is widely used in healthcare to interpret predictions from complex models, particularly in cases like patient risk stratification and diagnosis [66].

A 2025 study on interpretable machine learning for thyroid cancer recurrence prediction demonstrated that SHAP analysis effectively identifies key predictive features, enhancing model transparency [66].

SHAP visualizations including summary plots, waterfall plots, and force plots are integrated into the Streamlit dashboard to help users understand which factors most influence a project's risk assessment. User Acceptance Testing (UAT) with 3-5 stakeholders evaluates whether model reasoning and feature importance rankings are clear and actionable.

## Components and Design

### System Architecture

*Figure 2: MAAGAP SYSTEM ARCHITECTURE*

The MAAGAP system follows a four-layer architecture designed for modularity and scalability, as illustrated in Figure 2. The **Data Layer** manages ingestion and storage of historical project records from government agencies such as the Planning and Development Office, DPWH, and DBM, supplemented by external feeds from PAGASA weather APIs and PSA economic indicators. SQLite serves as the development database, with PostgreSQL available for production deployment. The **Processing Layer** implements the core AI components: a Predictive Analytics Engine using Random Forest and Gradient Boosting classifiers via scikit-learn, along with LSTM networks for temporal pattern recognition; a Risk Scoring Engine that classifies projects into Low (0--0.3), Medium (0.3--0.7), and High (0.7--1.0) risk categories; and an Optimization Engine using PuLP for linear programming-based resource allocation. The **Application Layer** provides Flask-based REST APIs for data submission, risk queries, and recommendations, alongside alert management and authentication services. The **Presentation Layer** delivers an interactive Streamlit dashboard featuring risk heatmaps, project detail views, resource allocation visualizations, and automated PDF reports.

Data flows through the system from ingestion and preprocessing, through prediction and risk scoring, to optimization and dashboard presentation. A feedback loop records actual outcomes to continuously retrain models, ensuring improved accuracy over time.

### Software Architecture

*Figure 3: MAAGAP SOFTWARE ARCHITECTURE*

As illustrated in Figure 3, the MAAGAP system employs a modular, three-tier Client-Server architecture to ensure scalability and efficiently separate heavy computational tasks from the user interface. The **Presentation Tier** features an interactive Streamlit web dashboard that translates complex AI predictions into accessible visual formats, such as risk heatmaps and resource allocation charts, communicating seamlessly with the backend via REST API. This **Application Tier** operates on a Flask-based server that orchestrates the system's core logic through three specialized modules: a Predictive Analytics Engine utilizing ensemble ML models (Random Forest, Gradient Boosting, LSTM) to forecast delays; a Risk Scoring Engine to classify project threats; and a PuLP-driven Optimization Engine for constrained resource deployment. Finally, the **Data Tier** relies on a PostgreSQL database to securely store project records, ingest external contextual variables from PAGASA and PSA APIs, and maintain a continuous feedback loop of actual project outcomes to iteratively retrain and improve the machine learning models over time.

### Database Design

*Figure 4: MAAGAP ENTITY-RELATIONSHIP DIAGRAM (ERD)*

As illustrated in Figure 4, MAAGAP employs a PostgreSQL relational database to normalize the Planning and Development Office's (PPDO) flat-file records into a time-series-capable schema tailored for predictive modeling and manpower optimization.

The architecture is organized into key interconnected entities: the **Project Registry** (PROJECT and CONTRACTOR), which stores immutable metadata and supplier reliability history; **Progress Tracking** (INSPECTION_LOG and PPDO_INSPECTOR), which captures dynamic, timestamped data to compute the actual-versus-target project slippage essential for LSTM velocity forecasting; **Contextual Feature Integration** (EXTERNAL_CONTEXT), which ingests external variables like PAGASA weather data to feed the Gradient Boosting and Random Forest classifiers; and the **Analytics and Scheduling Output** (MAAGAP_PREDICTIONS), which logs the AI-generated risk tiers (Low, Medium, High, Critical) and PuLP-optimized deployment schedules to ensure limited PPDO inspectors are efficiently prioritized and dispatched to high-risk sites.

### Procedural Design

*Figure 5: MAAGAP PROCEDURAL DESIGN AND PROCESS FLOWCHART*

As depicted in Figure 5, the MAAGAP system operates through a structured, five-phase procedural workflow to transform raw project monitoring data into actionable prescriptive insights. In **Phase 1**, the system extracts baseline project metadata and historical logs while synchronizing with external PAGASA weather and PSA economic APIs. **Phase 2** focuses on data preprocessing and feature engineering, crucially calculating Project Slippage by comparing target versus actual physical accomplishments. This engineered feature set is then processed in **Phase 3** by ensemble machine learning models---Random Forest, Gradient Boosting, and LSTM networks---to forecast schedule delays and classify projects into discrete risk tiers (Low, Medium, High, or Critical). For elevated-risk projects, **Phase 4** triggers the PuLP Optimization Engine to solve a linear programming assignment problem, generating an optimized inspector deployment schedule strictly constrained by PPDO manpower availability. Finally, **Phase 5** pushes these outputs to an interactive Streamlit presentation dashboard and logs the actual inspection outcomes back into the database, establishing a continuous feedback loop for iterative model retraining.

### Object-Oriented Design

*Figure 6: MAAGAP USE CASE DIAGRAM*

As illustrated in Figure 6, the MAAGAP system's functional requirements and user interactions are defined through a UML Use Case model that delineates the boundaries of three human actors and one automated system actor within the Planning and Development Office (PPDO). The **PPDO Manager** exercises strategic oversight by utilizing the Streamlit dashboard to monitor real-time risk heatmaps, review resource reallocation recommendations, and generate automated PDF reports for transparency compliance. Operating on the ground, the **PPDO Inspector** accesses AI-optimized deployment schedules and logs the actual physical and financial accomplishments gathered during site visits. Backend operations are maintained by the **System Administrator**, who manages user credentials and batch-uploads legacy DPWH or DBM project datasets to initialize new monitoring cycles. Powering these user interactions is the automated **MAAGAP AI Engine**, which executes the core computational logic: calculating project slippage from inspector logs, triggering ensemble predictive models (Random Forest, Gradient Boosting, LSTM) to assess risk tiers, and running PuLP linear programming algorithms to generate strictly constrained manpower allocation schedules.

*Figure 7: MAAGAP CLASS DIAGRAM*

As illustrated in Figure 7, the MAAGAP system's object-oriented structure is defined through a UML Class Diagram that bridges the relational database layer with complex computational logic across four primary class groupings. The **Data Entity** classes act as in-memory representations of database records, comprising Project for calculating target progress, Contractor for assessing historical reliability, InspectionLog for computing actual-versus-target physical slippage, and Inspector for tracking field personnel availability. **Contextual and Record Integrations** are managed by the ExternalContext class, which ingests real-time PAGASA and PSA environmental variables, and the PredictionRecord class, which archives AI outputs to establish a continuous feedback loop for model retraining. The core processing logic resides within the **Analytical Engine** classes: the PredictiveEngine instantiates Random Forest and LSTM networks to forecast project delays, the RiskScoringEngine evaluates these forecasts against predefined thresholds to classify projects into discrete risk tiers, and the OptimizationEngine utilizes the PuLP solver to generate mathematically constrained field deployment schedules. Finally, the **System Controller** classes govern application interactions, utilizing the DashboardController to render Streamlit visual components like risk heatmaps, and the AuthService to secure sensitive government data through strict user authentication and permission management.

*Figure 8: MAAGAP STATE MACHINE DIAGRAM*

As conceptualized in Figure 8, the dynamic lifecycle of infrastructure projects within the MAAGAP framework is modeled through a UML State Machine Diagram that maps discrete states governed by field inputs, AI evaluations, and administrative realities. Projects initialize in the *Registered* state via batch ingestion and transition to *Active Monitoring*, remaining there until PPDO inspectors log actual physical and financial accomplishments, which triggers the *Analyzing Risk* state. In this state, the system calculates slippage, ingests PAGASA and PSA data, and processes these features through ensemble models (Random Forest, Gradient Boosting, LSTM) to branch the project into either a *Low-Risk State* (returning to routine monitoring) or an *Elevated-Risk State* (Medium, High, or Critical). Elevated projects enter *Pending Optimization*, where the PuLP linear programming engine formulates constrained manpower allocations, subsequently moving to *Intervention Scheduled* upon PPDO Manager approval before looping back to active monitoring. To account for real-world bottlenecks like right-of-way disputes or severe weather, active projects may transition to a *Suspended* state before resuming work or moving to a permanently *Terminated* state. Finally, upon reaching 100% accomplishment, projects enter the *Completed* state; crucially, the final timelines, cost outcomes, and root causes of both completed and terminated projects are permanently archived in the PostgreSQL database to feed a continuous feedback loop for iterative machine learning model retraining.

*Figure 9: MAAGAP DEPLOYMENT DIAGRAM*

As illustrated in Figure 9, the physical execution environment and distribution of software components for the MAAGAP system are mapped through a comprehensive UML Deployment Diagram utilizing a scalable, multi-node Client-Server architecture. The framework connects four distinct hardware nodes via secure network protocols, beginning with the **Client Environment Node**, which allows PPDO personnel to interact seamlessly through standard web browsers via HTTPS and batch-upload legacy DPWH/DBM records from local file systems. To prevent client-side bottlenecks, heavy processing is isolated within the **Application Server Node**; this core computational hub hosts the Streamlit presentation dashboard, Flask-based REST APIs, security and alert infrastructures, and the computationally intensive machine learning and PuLP optimization engines. Data integrity is strictly maintained by the **Database Server Nodes** through a dual-database strategy---utilizing SQLite for local development and PostgreSQL communicating via standard TCP/IP protocols for production data ingestion and the continuous model-retraining feedback loop. Finally, the architecture integrates with the broader technological ecosystem via the **External Government Data Providers Node**, where the Application Server initiates secure outward HTTPS requests to ingest supplementary real-time environmental and economic variables from third-party PAGASA and PSA RESTful APIs.

### Process Design

*Figure 10: MAAGAP DATA FLOW DIAGRAM LEVEL 0*

As mapped in Figure 10, the absolute boundary and high-level data interactions of the software are established through a Level 0 Data Flow Diagram (Context Diagram) that abstracts the entire MAAGAP architecture into a single, centralized "black box" node (Process 0.0). This centralized process isolates macro-level data exchanges across four primary external entities: **Partner Agencies** (such as DPWH and DBM) supply foundational inbound flows of project master lists, budgets, and target schedules, while **External APIs** continuously feed the predictive algorithms with real-time PAGASA weather conditions and PSA economic indicators. Operating bidirectionally, **Field Inspectors** submit critical inbound inspection logs and actual physical/financial accomplishments, receiving AI-optimized, constrained field deployment schedules in return. Ultimately, the system delivers high-value outbound intelligence---including predictive risk heatmaps, automated PDF reports, and resource reallocation recommendations---to **PPDO Management**, who subsequently close the operational loop by returning inbound schedule approvals and administrative adjustments.

*Figure 11: MAAGAP DATA FLOW DIAGRAM LEVEL 1*

As detailed in Figure 11, a Level 1 Data Flow Diagram deconstructs the MAAGAP system's internal operations into five sequential sub-processes that map the chronological maturation of raw inputs into predictive insights and prescriptive schedules. The workflow initiates with **Process 1.0** (Manage Project Registry), which formats historical agency master lists into the Project Database (D1), establishing the baseline for **Process 2.0** (Record Field Inspections) to calculate and log actual-versus-target project slippage into the Inspection Logs (D2). Serving as the core analytical hub, **Process 3.0** (Predict Risk & Delays) aggregates data from D1, D2, and external PAGASA/PSA APIs, feeding this integrated dataset into ensemble machine learning models to forecast delays and save actionable risk tiers to the ML Predictions (D3) data store. Subsequently, **Process 4.0** (Optimize Deployments) routes elevated-risk projects to a PuLP linear programming solver that queries the PPDO-maintained Inspector Roster (D5) to formulate strictly constrained manpower allocation strategies, outputting the results to Deployment Schedules (D4). Concluding the cycle, **Process 5.0** (Render Dashboard & Reports) retrieves these aggregated outputs to generate real-time risk heatmaps and PDF reports for the PPDO Manager, while distributing the newly prioritized inspection routes back to the field personnel.

*Figure 12: MAAGAP DATA FLOW DIAGRAM LEVEL 2 (PREDICTION ENGINE)*

As detailed in Figure 12, a Level 2 Data Flow Diagram provides a granular view of the system's core artificial intelligence mechanics by deconstructing Process 3.0 (Predict Risk & Delays) into five specific sub-processes. The sequence begins with **Process 3.1** extracting static metadata from the Project Database (D1) and dynamic variance from Inspection Logs (D2) to structure the time-series arrays required for LSTM networks, while **Process 3.2** operates in parallel to ingest and normalize contextual PAGASA weather and PSA economic data. These engineered features are then consumed by **Process 3.3**, where ensemble machine learning models---specifically Random Forest and Gradient Boosting---generate quantitative forecasts for schedule delays and budget overruns. Subsequently, **Process 3.4** evaluates these raw probabilities against predefined thresholds to classify each project into an actionable risk tier (Low, Medium, High, or Critical) within the ML Predictions (D3) data store. Concurrently, to ensure public sector accountability, **Process 3.5** utilizes Explainable AI (XAI) principles to extract the underlying decision-making logic and weighted variables from the models, archiving this transparent reasoning in D3 for PPDO Manager review.

*Figure 13: MAAGAP DATA FLOW DIAGRAM LEVEL 2 (OPTIMIZATION ENGINE)*

As depicted in Figure 13, a second Level 2 Data Flow Diagram details the system's prescriptive capabilities by deconstructing Process 4.0 (Optimize Deployments) into five sequential sub-processes that transform AI risk classifications into actionable, mathematically constrained field schedules. The flow initiates with **Process 4.1** querying the ML Predictions (D3) data store to filter and isolate elevated-risk projects (Medium, High, Critical), while **Process 4.2** concurrently retrieves real-time personnel constraints, such as availability and workload capacities, from the Inspector Roster (D5). These inputs are synthesized in **Process 4.3** to formulate a formal linear programming mathematical model, defining an objective function to maximize high-risk project monitoring against the strict manpower limitations. This model is then fed into **Process 4.4**, where the PuLP solver acts as the computational core to process the assignment variables and generate an optimal assignment matrix. Finally, **Process 4.5** translates this raw mathematical matrix into a readable, prioritized routing schedule, saving the finalized assignments to the Deployment Schedules (D4) data store for immediate distribution to the presentation dashboard and field inspectors.

## Methodology

### System Development Life Cycle

*Figure 14: ITERATIVE SDLC PROCESS*

MAAGAP follows an agile, iterative development process executed over four phases spanning 8-10 months:

**Phase 1: Foundation and Data Preparation (Months 1-2)**

This phase establishes partnerships with local government agencies, particularly the Planning and Development Office of Iloilo City or nearby municipalities. These agencies maintain comprehensive records of infrastructure projects including timelines, budget allocations, contractor information, and completion status. Data preprocessing involves cleaning, normalization, handling missing values, and exploratory data analysis to identify patterns and inform feature selection. Should agency partnerships prove unfeasible, synthetic datasets are generated using realistic parameters documented in literature and validated against publicly available project summaries from government transparency portals.

**Phase 2: Model Development (Months 3-5)**

This phase focuses on developing and training the predictive analytics components. Ensemble machine learning methods (Random Forest, Gradient Boosting) are implemented via scikit-learn. LSTM networks for sequential pattern recognition are developed using TensorFlow/Keras. Model training uses 70/15/15 train-validation-test splits with performance evaluated using accuracy, precision, recall, F1-score, and MAE. The research prioritizes functional prediction over exhaustive hyperparameter optimization, recognizing the constraints of student-led research.

**Phase 3: Optimization and Dashboard Development (Months 6-8)**

Resource allocation optimization is formulated as a linear programming problem solved using PuLP. The web-based dashboard is developed using Flask for REST APIs and Streamlit for the interactive interface, chosen for their lightweight architecture and rapid development capabilities. The dashboard visualizes risk predictions, resource allocation recommendations, and project portfolio analytics in accessible formats for non-technical decision-makers. Features include risk heatmaps, project detail views, resource allocation visualizations, and automated PDF reports.

**Phase 4: Evaluation and Documentation (Months 9-10)**

System performance is assessed through quantitative evaluation (accuracy metrics on held-out test data), qualitative evaluation (user feedback sessions with faculty advisors or local government representatives), and comparative analysis (benchmarking predictions against historical outcomes). Functional testing confirms the system's ability to process batch-uploaded datasets and generate automated risk heatmaps and PDF reports without errors. Deliverables include a working prototype, comprehensive technical documentation, user guides, and thesis manuscript.

### Ethical Considerations and Data Privacy

The research adheres to ethical guidelines for handling sensitive government data. All project data obtained from partner agencies are anonymized, removing identifying information about specific contractors, officials, or politically sensitive details. Data is stored securely with access limited to research team members. Published results present aggregate findings rather than individual project details.

Should synthetic data be necessary, generation methods are documented transparently to enable scrutiny of assumptions and parameters. The team commits to responsible AI principles, ensuring prediction models do not perpetuate biases against particular contractors, regions, or project types. All code and methodology documentation are made available for peer review upon thesis completion, promoting transparency and reproducibility in AI research for public sector applications.

---

# References

[1] S. S. Almalki, "AI-Driven Decision Support Systems in Agile Software Project Management: Enhancing Risk Mitigation and Resource Allocation," *Systems*, vol. 13, no. 3, p. 208, Mar. 2025. doi: 10.3390/systems13030208

[2] S. Alsulamy, "Comparative analysis of deep learning algorithms for predicting construction project delays in Saudi Arabia," *Applied Soft Computing*, vol. 112, p. 112890, Jan. 2025. doi: 10.1016/j.asoc.2024.112890

[3] "Applying Predictive Analytics in Project Planning to Improve Task Estimation, Resource Allocation, and Delivery Accuracy," ResearchGate, Mar. 01, 2026. [Online]. Available: https://www.researchgate.net/publication/393864637_Applying_Predictive_Analytics_in_Project_Planning_to_Improve_Task_Estimation_Resource_Allocation_and_Delivery_Accuracy

[4] "Artificial Intelligence and Machine Learning in Project Management: A Conceptual Framework for Future Integration," *Journal of Business and Management Studies*, Al-Kindi Center for Research and Development, Mar. 01, 2026. [Online]. Available: https://al-kindipublishers.org/index.php/jbms/article/view/10776

[5] Asian Development Bank, *Climate Risk and Adaptation in the Philippines: Building Resilience in Infrastructure Projects*. Manila: ADB Southeast Asia Department, 2021.

[6] Bangsamoro Information Office, "BARMM regional statistics committee strengthens capacity on mastering data management," Bangsamoro Official Website, Apr. 29, 2024. [Online]. Available: https://bangsamoro.gov.ph/news/latest-news/barmm-regional-statistics-committee-strengthens-capacity-on-mastering-data-management/. [Accessed: Feb. 27, 2026].

[7] L. G. Banua, "Enhanced RPMES on-line," Unpublished master's thesis, Public Management Development Program, Development Academy of the Philippines, 2012. [Online]. Available: https://library.dap.edu.ph/cgi-bin/koha/opac-detail.pl?biblionumber=2464. [Accessed: Feb. 27, 2026].

[8] Center for Disaster Preparedness, "Inclusive Data Management System Guidebook," cdp.org.ph. [Online]. Available: https://www.cdp.org.ph/product-page/inclusive-data-management-system-guidebook. [Accessed: Feb. 27, 2026].

[9] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785-794.

[10] D. Choo et al., "Optimizing Health Coverage in Ethiopia: A Learning-Augmented Approach and Persistent Proportionality Under an Online Budget," arXiv preprint arXiv:2509.00135, Aug. 2025. [Online]. Available: https://arxiv.org/abs/2509.00135

[11] CIMdata, Inc., "ALICE Technologies Introduces New Feature: Schedule Insights Agent," cimdata.com, Oct. 1, 2025. [Online]. Available: https://www.cimdata.com/de/industry-summary-articles/item/28515-alice-technologies-introduces-new-feature-schedule-insights-agent. [Accessed: Mar. 1, 2026].

[12] CIMdata, Inc., "ALICE Technologies Launches New Visual Planning Product: ALICE Plan," cimdata.com, Apr. 25, 2025. [Online]. Available: https://www.cimdata.com/zh/industry-summary-articles/item/27474-alice-technologies-launches-new-visual-planning-product-alice-plan. [Accessed: Mar. 1, 2026].

[13] CIMdata, Inc., "Zachry Construction Corporation Implements ALICE to Drive Innovation in Heavy Civil Construction," cimdata.com, Jan. 23, 2025. [Online]. Available: https://www.cimdata.com/en/industry-summary-articles/item/26675-zachry-construction-corporation-implements-alice-to-drive-innovation-in-heavy-civil-construction. [Accessed: Mar. 1, 2026].

[14] City Government of Naga, "Ordinance No. 2025-061: An Ordinance Institutionalizing Quality Assurance of Government Infrastructure Projects in Naga City," Aug. 27, 2025. [Online]. Available: https://www2.naga.gov.ph/sp_ordinances/ordinance-no-2025-061/. [Accessed: Feb. 27, 2026].

[15] Commission on Audit, *Annual Audit Report on National Government Agencies and Local Government Units*. Quezon City: Republic of the Philippines, 2024.

[16] Commission on Audit, "Information and Publicity on Programs/Projects/Activities of Government Agencies," coa.gov.ph. [Online]. Available: https://coa.gov.ph/wp-content/uploads/ABC-Help/Updated_Guidelines_in_the_Audit_of_Procurement/annex%209/annex9-1.htm. [Accessed: Feb. 26, 2026].

[17] Department of Budget and Management, "DBM, UP-NOAH, DPWH sign agreement to enhance system for monitoring, evaluation of big-ticket projects," dbm.gov.ph, May 20, 2025. [Online]. Available: https://www.dbm.gov.ph/index.php/management-2/3375-dbm-up-noah-dpwh-sign-agreement-to-enhance-system-for-monitoring-evaluation-of-big-ticket-projects. [Accessed: Feb. 27, 2026].

[18] "Dynamic Workload Management System in the Public Sector: A Comparative Analysis," *Future Internet*, vol. 17, no. 3, p. 119, Mar. 2025. doi: 10.3390/fi17030119

[19] C. N. Egwim, H. Alaka, L. O. Toriola-Coker, H. Balogun, and A. Ajayi, "Applied artificial intelligence for predicting construction projects delay," *Machine Learning with Applications*, vol. 6, p. 100166, Dec. 2021. doi: 10.1016/j.mlwa.2021.100166

[20] C. N. Egwim, H. Alaka, L. O. Toriola-Coker, H. Balogun, and F. Sumnola, "Systematic review of critical drivers for delay risk prediction: Towards a conceptual framework for BIM-based construction projects," *Frontiers in Engineering and Built Environment*, vol. 3, no. 1, pp. 16-31, 2022. doi: 10.1108/FEBE-05-2022-0017

[21] D. J. Esguerra, "DBM launches online tracker for flood control projects," *Philippine News Agency*, Aug. 28, 2025. [Online]. Available: https://www.pna.gov.ph/index.php/articles/1257469. [Accessed: Feb. 27, 2026].

[22] D. J. Esguerra, "'Full disclosure': Marcos unveils AI-powered infra transparency portal," *Philippine News Agency*, Nov. 24, 2025. [Online]. Available: https://www.pna.gov.ph/index.php/articles/1263861. [Accessed: Feb. 26, 2026].

[23] D. J. Esguerra, "Gov't unveils digital tracker to monitor, evaluate big-ticket projects," *Philippine News Agency*, May 21, 2025. [Online]. Available: https://www.pna.gov.ph/articles/1250505. [Accessed: Feb. 27, 2026].

[24] D. J. Esguerra, "'Old data': DOH shrugs off COA report on P405-M idle medical gear," *Philippine News Agency*, Jan. 07, 2026. [Online]. Available: https://www.pna.gov.ph/articles/1266354

[25] European Space Agency, "SAFER," esa.int, Jun. 16, 2025. [Online]. Available: https://business.esa.int/projects/safer. [Accessed: Mar. 1, 2026].

[26] Fast and Efficient Budget Execution | DBM, accessed March 1, 2026. [Online]. Available: https://www.dbm.gov.ph/wp-content/uploads/Executive%20Summary/Fast%20and%20Efficient%20Budget%20Execution%20(updated%20as%20of%2007042016).pdf

[27] Forecast, "Official Website," forecast.app, 2026. [Online]. Available: https://forecast.app/. [Accessed: Mar. 1, 2026].

[28] Forecast, "Overview of Artificial Intelligence (AI) Features," forecast.app, Oct. 13, 2025. [Online]. Available: https://support.forecast.app/hc/en-us/articles/39075249944849-Overview-of-Artificial-Intelligence-AI-Features. [Accessed: Mar. 1, 2026].

[29] "From Data to Decisions: How IT Projects Leverage Forecasting for Resource Allocation," Cogent Info, Mar. 01, 2026. [Online]. Available: https://cogentinfo.com/resources/from-data-to-decisions-how-it-projects-leverage-forecasting-for-resource-allocation

[30] Get It Right Initiative, "The use of technology to reduce errors in design and construction: A best practice casebook," getitright.uk.com, 2024. [Online]. Available: https://getitright.uk.com/reports/giri-research-report-the-use-of-technology-to-reduce-errors-in-design-and-construction-a-best-practice-casebook [Accessed: Mar. 1, 2026].

[31] S. Ghadekar et al., "Unveiling Explainable AI in Healthcare: Current Trends, Challenges, and Future Directions," *WIREs Data Mining and Knowledge Discovery*, 2025. https://doi.org/10.1002/widm.70018

[32] R. A. Gita-Carlos, "PCO urges public to monitor infra projects via transparency portal," *Philippine News Agency*, Nov. 25, 2025. [Online]. Available: https://www.pna.gov.ph/articles/1263908. [Accessed: Feb. 26, 2026].

[33] R. A. Gita-Carlos, "PS-DBM PhilGEPS eMarketplace launched," *Philippine News Agency*, Dec. 14, 2024. [Online]. Available: https://www.pna.gov.ph/articles/1239957. [Accessed: Feb. 26, 2026].

[34] A. Gondia, A. Siam, W. El-Dakhakhni, and A. H. Nassar, "Machine learning algorithms for construction projects delay risk prediction," *Journal of Construction Engineering and Management*, vol. 146, no. 1, p. 04019085, Jan. 2020. doi: 10.1061/(ASCE)CO.1943-7862.0001736

[35] Governance DepDev Portal, "Gov't unveils digital tracker to monitor, evaluate big-ticket projects," governance.depdev.gov.ph, May 21, 2025. [Online]. Available: https://governance.depdev.gov.ph/govt-unveils-digital-tracker-to-monitor-evaluate-big-ticket-projects/. [Accessed: Feb. 27, 2026].

[36] A. R. Hevner, S. T. March, J. Park, and S. Ram, "Design science in information systems research," *MIS Quarterly*, vol. 28, no. 1, pp. 75-105, 2004.

[37] "Implementation of Machine Learning-Based Risk Prediction Models for Large-Scale Infrastructure Construction Projects in Urban Environments," BRAJETS, Mar. 2025. [Online]. Available: https://brajets.com/brajets/article/view/2099

[38] "Leveraging Artificial Intelligence and Data Analytics for Decision-Making in IT Project Management," PwC CEE IT Practice Technical Report, CORE Repository, Jun. 2025. [Online]. Available: https://core.ac.uk/outputs/658974946/

[39] Y. Li et al., "Multi-objective optimization of industrial productivity and renewable energy allocation based on NSGA-II for carbon reduction and cost efficiency," *Energies*, vol. 18, no. 20, p. 5438, 2025. https://doi.org/10.3390/en18205438

[40] Y. Liu et al., "Multi-objective optimization method for building energy-efficient design based on multi-agent-assisted NSGA-II," *Energy Informatics*, vol. 7, no. 1, pp. 1-20, 2024. https://doi.org/10.1186/s42162-024-00394-4

[41] Y. Liu et al., "NSGA-II based multi-objective disaster recovery scheduling for virtual cloud platforms," *Informatica*, vol. 36, no. 4, pp. 11126, 2025. http://www.informatica.si/index.php/informatica/article/view/11126

[42] S. M. Lundberg and S. I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 4765-4774.

[43] Y. Ma et al., "An improved NSGA-II based on multi-task optimization for multi-UAV maritime search and rescue under severe weather," *Journal of Marine Science and Engineering*, vol. 11, no. 4, p. 781, 2023.

[44] "Machine Learning-based Feature Evaluation of the Factors Affecting the Effective Infrastructure Project Implementation," ResearchGate, Nov. 2025. [Online]. Available: https://www.researchgate.net/publication/397958656

[45] E. Marcelo, "DOH flagged for P405 million idle medical equipment," *The Philippine Star*, Jan. 06, 2026. [Online]. Available: https://www.philstar.com/headlines/2026/01/06/2499036/doh-flagged-p405-million-idle-medical-equipment

[46] M. S. A. Engineering, "Advanced Machine Learning Models for Predicting Project Performance in Complex Construction Environments," ResearchGate, Jun. 2025. [Online]. Available: https://www.researchgate.net/publication/393204693

[47] F. Mostofi, O. B. Tokdemir, and V. Toğan, "Construction Delay Prediction Model Using a Relationship-Aware Multihead Graph Attention Network," *Journal of Management in Engineering - ASCE*, vol. 41, no. 3, p. 04025010, May 2025. doi: 10.1061/JMENEA.MEENG-6245

[48] Oracle Corporation, "Oracle Construction and Engineering Intelligence," oracle.com, 2025. [Online]. Available: https://www.oracle.com/cn/construction-engineering/intelligence/. [Accessed: Mar. 1, 2026].

[49] Oracle Corporation, "Oracle Construction Intelligence Cloud, Datasheet," oracle.com, 2024. [Online]. Available: https://www.oracle.com/tr/construction-engineering/construction-intelligence-cloud/datasheet/. [Accessed: Mar. 1, 2026].

[50] J. C. Paunan, "Marcos unveils AI-powered portal to track public works, fight corruption," *Philippine Information Agency*, Nov. 24, 2025. [Online]. Available: https://pia.gov.ph/news/marcos-unveils-ai-powered-portal-to-track-public-works-fight-corruption/. [Accessed: Feb. 26, 2026].

[51] K. Peffers, T. Tuunanen, M. A. Rothenberger, and S. Chatterjee, "A design science research methodology for information systems research," *Journal of Management Information Systems*, vol. 24, no. 3, pp. 45-77, 2007.

[52] Philippine Government Electronic Procurement System, "PhilGEPS Official Portal," philgeps.gov.ph. [Accessed: Feb. 26, 2026].

[53] Philippine Space Agency, "PhilSA, DPWH to use space data for infrastructure monitoring," philsa.gov.ph, Nov. 5, 2025. [Online]. Available: https://philsa.gov.ph/news/philsa-dpwh-to-use-space-data-for-infrastructure-monitoring/. [Accessed: Feb. 26, 2026].

[54] Philippine Statistical Research and Training Institute, "Sandiganbayan Batch 2 Deepens Data Management Skills with Excel and Google Sheets Training," psrti.gov.ph, Aug. 2025. [Online]. Available: https://psrti.gov.ph/sandiganbayan-batch-2-deepens-data-management-skills-with-excel-and-google-sheets-training/. [Accessed: Feb. 27, 2026].

[55] Philstar Global, "P6.5 billion worth of DPWH projects in 2024 'unusable' --- COA," Philstar.com, Dec. 04, 2025. [Online]. Available: https://www.philstar.com/headlines/2025/12/04/2491664/coa-finds-747-unusable-idle-dpwh-projects-2024

[56] L. Potin, R. Figueiredo, V. Labatut, and C. Largeron, "Pattern mining for anomaly detection in graphs: Application to fraud in public procurement," in *ECML PKDD 2023*, 2023. [Online]. Available: arXiv:2306.10857. DOI: 10.1007/978-3-031-43427-3_5.

[57] "Predictive Risk Intelligence: Staying Ahead of Tomorrow's Challenges," MitKat Advisory, Mar. 01, 2026. [Online]. Available: https://mitkatadvisory.com/predictive-risk-intelligence-staying-ahead-of-tomorrows-challenges/

[58] "Predictive risk assessment: Empower your security strategy in 2026," TrustCloud, Mar. 01, 2026. [Online]. Available: https://www.trustcloud.ai/risk-management/predictive-risk-assessment-preventing-security-incidents/

[59] Procurement Service -- Department of Budget and Management, "PS-DBM conducts training with BAI on the use of modernized PhilGEPS," PS-PhilGEPS, Apr. 7, 2025. [Online]. Available: https://ps-philgeps.gov.ph/home/index.php/about-ps/news/7480-ps-dbm-conducts-training-with-bai-on-the-use-of-modernized-philgeps. [Accessed: Feb. 26, 2026].

[60] PTV News, "Gov't, U.P. sign MOU on digital monitoring, evaluation of big-ticket projects," ptvnews.ph, May 21, 2025. [Online]. Available: https://ptvnews.ph/govt-u-p-sign-mou-on-digital-monitoring-evaluation-of-big-ticket-projects/. [Accessed: Feb. 27, 2026].

[61] K. Ransikarbum and S. J. Mason, "A bi-objective optimisation of post-disaster relief distribution and short-term network restoration using hybrid NSGA-II algorithm," *International Journal of Production Research*, vol. 60, no. 19, pp. 5769-5793, 2022.

[62] A. G. L. Romme, "Design science as experimental methodology in innovation and entrepreneurship research: A primer," *CERN IdeaSquare Journal of Experimental Innovation*, vol. 7, no. 2, pp. 4-7, 2023. https://doi.org/10.23726/cij.2022.1427

[63] P. Sahu et al., "Smart delay prediction: Supervised machine learning solutions for construction projects," *Journal of Mechanics of Continua and Mathematical Sciences*, vol. 20, no. 6, 2025. https://doi.org/10.26782/jmcms.2025.06.00010

[63] P. Sahu, D. K. Bera, P. K. Parhi, and M. Kandpal, "Smart delay prediction: Supervised machine learning solutions for construction projects," *Journal of Mechanics of Continua and Mathematical Sciences*, vol. 20, no. 6, pp. 138-152, Jun. 2025. doi: 10.26782/jmcms.2025.06.00010

[64] M. O. Sanni-Anibire, R. M. Zin, and S. O. Olatunji, "Machine learning-based framework for construction delay mitigation," *Journal of Information Technology in Construction*, vol. 26, pp. 139-155, 2021. doi: 10.36680/j.itcon.2021.017

[65] M. O. Sanni-Anibire, R. Mohamad Zin, and S. O. Olatunji, "Machine learning model for delay risk assessment in tall building projects," *International Journal of Construction Management*, vol. 22, no. 11, pp. 2134-2143, 2022. doi: 10.1080/15623599.2020.1768326

[66] A. Schindele et al., "Interpretable Machine Learning for Thyroid Cancer Recurrence Prediction: Leveraging XGBoost and SHAP Analysis," *European Journal of Radiology*, vol. 186, p. 112049, 2025. https://doi.org/10.1016/j.ejrad.2025.112049

[67] K. D. Strang, "Exploring IT project performance from government big data using supervised machine learning: a managerial perspective," *International Journal of Business Performance Management*, vol. 26, no. 5, pp. 660-685, 2025. doi: 10.1504/IJBPM.2025.148219

[68] B. Taha, A. H. Ibrahim, and A. A. Soliman, "Risk-indexed artificial neural network for predicting duration and cost of irrigation canal-lining projects using survey-based calibration and python validation," *Scientific Reports*, vol. 15, no. 1, p. 40316, Nov. 2025. doi: 10.1038/s41598-025-24125-1

[69] C. L. Tamayo and J. J. F. Famadico, "Predictive Models of Construction Project Success Rating Using Regression and Artificial Neural Network," *Philippine E-Journals*, 2024. [Online]. Available: https://ejournals.ph/article.php?id=24325

[70] "The Power of Machine Learning in Public Administration," The Training Data Project, Mar. 01, 2026. [Online]. Available: https://www.trainingdataproject.org/tdp-blog/the-power-of-machine-learning-in-public-administration

[71] "The role of artificial intelligence in the digital transformation of government: opportunities and ethical challenges," PMC, Mar. 01, 2026. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC12623404/

[72] Unanet, "Unanet's Product Innovations Advance Intelligence and Automation for GovCon and AEC Firms," unanet.com, Dec. 9, 2025. [Online]. Available: https://unanet.com/news/unanets-product-innovations-advance-intelligence-and-automation-for-govcon-and-aec-firms. [Accessed: Mar. 1, 2026].

[73] University of Edinburgh Bayes Centre, "Sharktower," ed.ac.uk, Sep. 30, 2024. [Online]. Available: https://bayes-centre.ed.ac.uk/accelerating-entrepreneurship/ai-accelerator/ai-accelerator-alumni/post-covid-ai-accelerator/sharktower. [Accessed: Mar. 1, 2026].

[74] "What is Explainable AI (XAI)?," IBM, Mar. 01, 2026. [Online]. Available: https://www.ibm.com/think/topics/explainable-ai

[75] World Bank, *Philippines Infrastructure Sector Assessment Program*. Washington, DC: World Bank Group, 2023.

[76] Z. M. Yaseen, Z. H. Ali, S. Q. Salih, and N. Al-Ansari, "Prediction of risk delay in construction projects using a hybrid artificial intelligence model," *Sustainability*, vol. 12, no. 4, p. 1514, Feb. 2020. doi: 10.3390/su12041514

[77] J. Yoders, "New Cloud-Based PM Capabilities Introduced by Oracle, Procore," *Engineering News-Record*, Jul. 18, 2022. [Online]. Available: https://www.enr.com/articles/54467-new-cloud-based-pm-capabilities-introduced-by-oracle-procore. [Accessed: Mar. 1, 2026].

[78] J. Yoders, "Oracle Buys Newmetrix to Integrate AI Tech Into Cloud Platform," *Engineering News-Record*, Dec. 16, 2022. [Online]. Available: https://www.enr.com/articles/55605-oracle-buys-newmetrix-to-integrate-ai-tech-into-cloud-platform. [Accessed: Mar. 1, 2026].

[1] Placeholder Image | Elementor Developers. [Online]. Available: https://developers.elementor.com/docs/hooks/placeholder-image/. [Accessed: Jan. 15, 2024].

---

# Appendices

## Appendix A Sample Appendix

## Appendix B Disclaimer

This research project and its corresponding documentation entitled "[Title Here]" is submitted to the College of Information and Communications Technology, West Visayas State University, in partial fulfillment of the requirements for the degree, [Degree Program]. It is the product of our own work, except where indicated text.

We hereby grant the College of Information and Communications Technology permission to freely use, publish in local or international journal/conferences, reproduce, or distribute publicly the paper and electronic copies of this software project and its corresponding documentation in whole or in part, provided that we are acknowledged.

[Author Firstname MI. Lastname]

[Author Firstname MI. Lastname]

[Author Firstname MI. Lastname]

[Author Firstname MI. Lastname]

[Author Firstname MI. Lastname]

June 2027
