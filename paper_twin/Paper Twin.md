# TWIN: Personalized Clinical Trial Digital Twin Generation

**Trisha Das1**  
Department of Computer Science,  
University of Illinois  
Urbana-Champaign  
Urbana, IL, USA  
trishad2@illinois.edu  

**Zifeng Wang1**  
Department of Computer Science,  
University of Illinois  
Urbana-Champaign  
Urbana, IL, USA  
zifengw2@illinois.edu  

**Jimeng Sun**  
Department of Computer Science,  
University of Illinois  
Urbana-Champaign  
Urbana, IL, USA  
jimeng@illinois.edu  

2023  

---

## Abstract  
Clinical trial digital twins are virtual patients that reflect personal characteristics in a high degree of granularity and can be used to simulate various patient outcomes under different conditions. With the growth of clinical trial databases captured by Electronic Data Capture (EDC) systems, there is a growing interest in using machine learning models to generate digital twins. This can benefit the drug development process by reducing the sample size required for participant recruitment, improving patient outcome predictive modeling, and mitigating privacy risks when sharing synthetic clinical trial data. However, prior research has mainly focused on generating Electronic Healthcare Records (EHRs), which often assume large training data and do not account for personalized synthetic patient record generation. In this paper, we propose a sample-efficient method TWIN for generating personalized clinical trial digital twins. TWIN can produce digital twins of patient-level clinical trial records with high fidelity to the targeting participant's record and preserves the temporal relations across visits and events. We compare our method with various baselines for generating real-world patient-level clinical trial data. The results show that TWIN generates synthetic trial data with high fidelity to facilitate patient outcome predictions in low-data scenarios and strong privacy protection against real patients from the trials.  

**Keywords**: Digital twin, Synthetic data, Clinical trial  

---

## Figure 1  
![Figure 1](Figure1.png)  
*An illustrative example of the process of generating personalized clinical trial digital twins based on the real follow-up visits recorded in real clinical trial data.*  

---

## 1. Introduction  
Clinical trials are prospective studies that aim to compare the effects and value of new interventions in human subjects. They typically recruit tens to hundreds of participants and can take several years to complete. With the rapid expansion of clinical trial data, there is a growing interest in using *Digital Twins* to simulate patient outcomes. Digital twins are virtual patients that reflect personal characteristics in a high degree of granularity and can be used to simulate various patient outcomes under different conditions. The use of digital twins in clinical trials offers several benefits, such as  

- Reducing the sample size required for participant recruitment by using predictive modeling to obtain outcomes for each patient under different arms, thus accelerating the drug development process.  
- Generating synthetic patient records to mitigate privacy risks when sharing real clinical trial data.  

The closest analog to the generation of digital twins for clinical trials is the generation of electronic healthcare records (EHRs). In particular, deep generative models such as generative adversarial networks (GANs) and variational autoencoders (VAEs) have been used to generate synthetic EHRs while preserving the statistical patterns of real EHRs (Bernard et al., 2017; Li et al., 2018; Zhang et al., 2019; Liu et al., 2020). These synthesized EHRs can then be used to develop health risk predictive models. However, there are several key differences between the generation of digital twins for trials and the generation of synthetic EHRs:  

- The structure and temporal patterns of clinical trial data differ substantially from those of EHRs (Bernard et al., 2017). Clinical trial data is less sparse and more regular in terms of event incidence patterns and intervals and has a much smaller sample size, which limits the use of large models. Hence it is suboptimal to apply EHR generation methods to clinical trial data.  
- The main objective of generating synthetic EHRs is to capture the global characteristics of real EHR data, whereas the goal of generating digital twins is to create virtual patients that fit personal characteristics with high fidelity, utility, and diversity.  

To be specific, the process of generating clinical trial digital twins can be broken down into three main steps, as illustrated in Fig. 1. Step one involves collecting the participant's baseline data at the time of admission. In step two, the participant's follow-up visits are recorded, and a machine learning model is used to simulate virtual visits based on the real visits and the patient's characteristics. Finally, in step three, personalized digital twins are generated based on the real participant record. By adjusting inputs such as the assigned treatment to the participant, it is possible to simulate the patient's trajectory in counterfactual scenarios.  

Concretely, we propose a framework for personalized clinical Trial digital **tWIN** generation (TWIN). Our main contributions are summarized as follows.  

- *To the best of our knowledge, we are the first to concentrate on personalized trial digital twin generation, whereas previous works only consider generating synthetic clinical trial data that are aligned with the real data in global statistics.*  
- *We propose a generative model that produces digital twins with high fidelity to mimic real participants, utilizes information from most similar participants, and preserves the causality across visits and events.*  
- *Our model can also simulate a patient's probable trajectory in a counterfactual scenario, i.e., if the patient were assigned to a different arm of the trial.*  

The rest of this paper is organized as follows: In §2, we review the related work in the literature. In §3, we dive into the proposed framework in detail. In §4, we present the results of our experiments. Finally, in §5, we provide a conclusion and discuss the directions for future work.  

---

## 2. Related Work  

### 2.1 Patient Outcome Prediction  
The rapid expansion of EHR data from millions of patients has rendered a profound impact on the data analytic modeling paradigm in healthcare (Kumar et al., 2019). Deep learning has been widely employed to encode the longitudinal patient records and then predict the patient health risk (Berger et al., 2017) or the disease progression (Liu et al., 2018). There are two main categories of EHR predictive modeling tasks studied: risk detection and sequential prediction of clinical events (Kumar et al., 2019). Risk detection seeks to predict the risk of the target event, e.g., disease onset, taking EHRs as inputs. The risk detection is formed in three types: First, as either binary classification (e.g., the onset of a specific disease (Berger et al., 2017; Li et al., 2018; Kummel et al., 2018) or mortality (Miller et al., 2016)); Second, as multi-class classification (e.g., disease stage prediction (Berger et al., 2017) and disease categories (Miller et al., 2016)); Third, as multi-label classification (e.g., diagnosis code assignment (Berger et al., 2017)). Sequential event prediction seeks to make a multi-label classification for multiple events at once (Berger et al., 2017; Li et al., 2018; Kummel et al., 2018; Miller et al., 2016).  

Patient predictive modeling is also the core to developing personalized medicine that identifies differences in treatment response and incidence rate of adverse effects based on individuals in clinical trials (Miller et al., 2016). However, the area of clinical trial predictive modeling has been less explored than EHRs. Most works focus on the predictive modeling of specific diseases (Li et al., 2018; Kummel et al., 2018; Miller et al., 2016; Mili et al., 2018) or of tabular clinical trial data (Miller et al., 2016). Directly adopting EHR sequence modeling to clinical trial data would yield suboptimal results due to the limited size of the data. One reason is the nature of clinical trials where each trial only recruits hundreds to thousands of participants at most. The other reason is that the privacy issue jeopardizes access to clinical trial data. Our work aims to handle these challenges in two aspects: augmenting clinical trial data to offer a sufficient amount of data for deep learning models and generating synthetic clinical trial data to be around the privacy risk of sharing real trial data.  

### 2.2 Synthetic Patient Record Generation  
Generating synthetic patient records is a promising solution to the privacy concerns that arise from the release of health-related data across institutions (Rosenblatt et al., 2000). Research in this field primarily focuses on generating synthetic EHRs using generative adversarial networks (GANs) (Berger et al., 2017; Li et al., 2018; Kummel et al., 2018; Mili et al., 2018), variational autoencoders (VAEs) (Li et al., 2018; Kummel et al., 2018; Mili et al., 2018), and language models (Miller et al., 2016). The goal of this line of research is to align global statistics as closely as possible with real EHRs. The generated synthetic data unlocks the collaboration across institutes on developing AI algorithms based on healthcare data, mitigating the concern of regulatory, intellectual property, and privacy barriers. Following the success of synthetic EHR generation, research has also begun to study synthetic clinical trial data generation (Rosenblatt et al., 2000), e.g., synthetic tabular clinical trial data (Berger et al., 2017; Li et al., 2018). Nevertheless, these efforts often ignore the temporal structure of clinical trial data, making them unable to replicate the original clinical trial structure or use them for longitudinal modeling.  

Besides, there were efforts committed to using probabilistic graphical models to fit the temporal distribution of clinical trial data on specific diseases like Alzheimer's Disease (Berger et al., 2017) and Multiple Sclerosis (Miller et al., 2016). Nevertheless, they concentrated on estimating the uncertainty of patient trajectories and the predicted endpoints. The utility of generated clinical trial data remains vague. In contrast, TWIN is capable of augmenting the volume of clinical trial data that support the employment of more sophisticated machine learning models for advanced prediction performance.  

---

## 3. Method  

This section covers the main framework of TWIN, including the problem definition, workflow, and training task formulation. The methods for creating synthetic clinical trial data are also discussed. Additionally, the process for evaluating TWIN’s effectiveness in terms of quality, utility, and privacy preservation is outlined.  

### 3.1 Problem Formulation  

In a clinical trial, there are \( N \) participants. The \( n^{\text{th}} \) patient is represented by a sequence of visits in the temporal order as  
\[ X_{n;1:T_n} = \{x_{n,1}, x_{n,2}, \cdots, x_{n,T_n}\}, \]  
where \( x_{n,t} \) denotes the events that occurred during the patient’s \( t^{\text{th}} \) visit and \( T_n \) is the total number of visits for that patient. Each visit \( x_{n,t} \) constitutes sets of events as  
\[ x_{n,t} = \{x_{n,t}^{1}, x_{n,t}^{2}, \cdots, x_{n,t}^{U}\}. \]  
(1)  


## Figure 2 
![Figure 2](Figure2.png) 

**Figure 2**: The workflow of data generation based on TWIN. (Right) TWIN takes the real follow-up visits \(X_{n,1:T_{n}}\) then generates the simulated visits \(\hat{X}_{n,1:T_{n}}\). (Left) The detailed working process of TWIN reconstructs the raw medication and adverse events with VAE, enhanced by retrieval-augmented encoding. An additional Causality Preserving Module (CPM) takes the latent event embeddings and treatments then make the cross-modality temporal predictions, i.e., the events that occurred in the next timestamp. *Treat.* is short for treatment; *Med* is short for medication; *AE* is short for adverse event.

where \(U\) is the total number of event types in the data. An event set \(x^{u}_{n,t}\) contains a set of events of type \(u\) occurred at the \(t^{\text{th}}\) visit of patient \(n\), denoted by \(x^{u}_{n,t}=\{c_{1},c_{2},\ldots,c_{l}\}\). Here, each \(c_{l}\) is an indicator \(c_{l}\in\{0,1\}\) showing if the event \(l\) occurred or not.

Without the loss of generality, we focus on the three major types of events: *treatment*, *medication*, and *adverse events*. Treatment events, represented by \(x^{1}_{n,t}\), are the assigned treatment at the \(t\)-th timestep for the participant \(n\). Note that for patient \(n\), the treatment plan must be in line with the arm she was assigned to, such as the treatment arm \(\mathcal{T}\) or control arm \(C\). This means that for all participants in the same arm, their treatment schedule should be the same across the entire tracking period, as \(\{x^{1}_{n,1},x^{1}_{n,2},\cdots,x^{1}_{n,T_{n}}\}\) are the same for \(n\in\mathcal{T}\). Here, \(\mathcal{T}\) is a set of participants' indices in the treatment arm, and \(C\) represents the control arm. This feature separates clinical trial data from EHRs because we should strictly maintain the treatment schedule when generating the digital twins for targeting participants, while EHR generation models do not have this constraint.

Medication events, represented by \(x^{2}_{n,t}\) or \(x^{med}_{n,t}\), are additional medications given to the patient that is not being examined in the trial. Adverse events, represented by \(x^{3}_{n,t}\) or \(x^{ae}_{n,t}\), are the unexpected adverse events observed in each visit. We argue it is vital to consider the *causality* across these two types of events when generating personalized digital twins. For instance, medication *Acetaminophen* at timestep \(t+1\) might be provided by the doctors due to the adverse event *fever* that happened at timestep t. On the other hand, the *fever* symptom may not be present at timestep \(t+1\) as the response to taking *Acetaminophen* at timestep \(t\). It is challenging for EHR generation models to fully capture this causality because they generate synthetic data step by step from \(t=0\) to \(t=T\), which renders an accumulation of errors because a slight event modification at early timesteps will yield a significant perturbation of the entire patient trajectory.

Concretely, this paper proposes to generate personalized clinical trial digital twins that preserve the *cross-event temporal causality* while generating digital twins: (1) Adverse events of visit \(t+1\) depend on the treatments and medications provided in visit \(t\). (2) Medications for visit \(t+1\) are provided based on the treatment and adverse events in visit \(t\). In the next, we will elaborate on the details of TWIN that abides by the causal dependencies and the constraints regarding treatment during the generation of digital twins.

### 3.2 Input Encoding

The raw patient record \(X_{n;1:T_{n}}\) is defined by Eq. (1), which contains a sequence of \(T_{n}\) visits recorded during the clinical trial. As illustrated by Fig. 1, we seek to generate the digital twin \(\hat{x}_{n;1:T_{n}}\) that retains the characteristics of the targeting patient while being in diverse trajectories. In addition, we urge the generated digital twin to be aligned with the cross-event temporal causality, i.e., the arbitrary two generated adjacent visits need to satisfy the causality constraints across timesteps and events.

Concretely, we propose a generator \(f(\cdot)\) that simulates personalized patient trajectories conditioned by the real record \(X_{n;1:T_{n}}\), the \(K\) nearest neighbor patients \(\{X_{k;1:T_{n}}\}_{k}\), and the learned model parameters \(\Theta\), as

\[
\hat{X}_{n;1:T_{n}}=f\left(X_{n;1:T_{n}},\{X_{k;1:T_{k}}\}_{k};\Theta\right).
\]  
(3)

**Event Encoding.** The raw input \(X_{n;1:T_{n}}\) contains a series of visits \(x_{n,t}\), where each visit \(x_{n,t}\) consists of concurrent events in \(U\) different types, as \(x_{n,t}=\{x^{1}_{n,t},\ldots,x^{U}_{n,t}\}\). One event set \(x^{u}_{n,t}\) is converted to a multi-hot vector where each element indicates the occurrence of a specific event at this timestep, as \(x^{u}_{n,t}=\{c_{1},\ldots,c_{l}\}\in\{0,1\}^{l}\). The raw input \(x^{u}_{n,t}\) can be encoded into a dense embedding by \(\mathbf{W}_{\text{emb}}^{u}\in\mathbb{R}^{l\times d}\) as

\[
\mathbf{h}_{n,t}^{u}=x^{u}_{n,t}\cdot\mathbf{W}_{\text{emb}}^{u}\in\mathbb{R}^d,
\]  
(4)

where \(\mathbf{W}_{\text{emb}}^{u}\) is a trainable embedding matrix; \(d\) is the hidden dimension; \(\hat{\mathbf{h}}_{n,t}^{u}\) is the embedding in *event level*. Similarly, we can encode for \(u\in\{1,\ldots,U\}\) using \(\{\mathbf{W}_{\text{emb}}^{u}\}_{u}^{U}\) to \(\{\mathbf{h}_{n,t}^{u}\}_{u}^{U}\).

**Retrieval-Augmented Encoding.** We propose to leverage an indexed retriever to augment the input encoding, i.e., retrieval-augmented encoding. In detail, for the targeting visit \(x_{n,t}\), we retrieve \(K\) visits \(\{x_{k,t^{\prime}}\}_{k}^{K}\) from all the other patients with the highest similarity to \(x_{n,t}\). Note that \(t^{\prime}\) can be either equal or unequal to \(t\). To keep the process efficient, we employ dot-product similarity for every pair \(\{x_{n,t},x_{k,t^{\prime}}\}\) based on the multi-hot encoded inputs and then rank to solicit the top \(K\) similar visits.

After that, we draw the events of type \(u\) from the retrieved results, as \(\{x_{k,t^{\prime}}^{u}\}_{k}^{K}\), and encode them to dense embeddings \(\{\mathbf{h}_{k,t^{\prime}}^{u}\}\) using \(\mathbf{W}_{\text{emb}}^{u}\). We further consider dynamically assigning importance weights for the retrieved results when fusing them using attention, as  

\[
\begin{split}
\tilde{\mathbf{h}}_{n,t}^{u} &= \text{Softmax}(x_{n,t} \cdot X_{n,K}^{\top}) \cdot \mathbf{H}_{n,K}^{u} = \sum_{k\in\mathcal{K},i\in \mathcal{I}} \alpha_{i}k, \\
&\text{where } \mathcal{K} = \{\mathbf{h}_{n,t}^{u}\} \cup \{\mathbf{h}_{k,t^{\prime}}^{u}\}_{k}^{K}, \mathcal{I} = \{1,\cdots,K+1\}.
\end{split}
\]  
(5)

And we have  

\[
X_{n,K} = [x_{n,t} \oplus \{x_{k,t^{\prime}}\}_{k}^{K}] \in \{0,1\}^{(K+1)\times \sum_{u}l_{u}}
\]  
(6)  

that is the concatenated input records. In particular, \(l_{u}\) is the total number of unique events of \(u\)-th modality, e.g., the number of unique medications; \(\alpha_{k}\) are affinity scores that measure the degree of similarity of the retrieved visits to the targeting visit; \(\mathbf{H}_{n,K}^{u} = [\mathbf{h}_{n,t}^{u} \oplus \{\mathbf{h}_{k,t^{\prime}}^{u}\}_{k}^{K}] \in \mathbb{R}^{(K+1)\times d}\) is the embedding matrix by concatenating all the encoded visit embeddings. In this way, the aggregated embedding \(\tilde{\mathbf{h}}_{n,t}^{u}\) incorporates the information from a larger pool of data and mitigates overfitting.


### 3.3 Digital Twin Generation  

With the encoded input events \(\tilde{\mathbf{h}}_{n,t}^{u}\), our method TWIN can perform two tasks for digital twin generation:  

**Personalized Generation**. The goal of this task is to generate synthetic patient records that are diverse and closely resemble the target patient records. This task allows personalized augmenting the existing patient records with synthetic records generated by the learned generative model.  

The digital twin generation tasks are supported by a Variational Auto-Encoder (VAE) that consists of an encoder \(q_{\phi}(\mathbf{z}|\mathbf{h})\) and a decoder \(p_{\theta}(x|\mathbf{z})\). For clarity, we use the abbreviated notation \(\mathbf{h}\) and \(x\) instead of \(\tilde{h}_{n,t}^{u}\) and \(x_{n,t}^{u}\), respectively. These two notations will be used interchangeably from this point forward. Formally, we parameterize the encoder \(q_{\phi}(\mathbf{z}|\mathbf{h})\) using Gaussian distribution, as \(q_{\phi}(\mathbf{z}|\mathbf{h})=\mathcal{N}(\mathbf{z}|\mu,\sigma\cdot I_{d})\). The sampling process is:  
1. Obtain the mean and variance of the Gaussian distribution from the encoder, i.e., \(\{\mu(\mathbf{h}),\sigma(\mathbf{h})\}\sim q_{\phi}(\mathbf{z}|\mathbf{h})\).  
2. Sample a random noise \(\epsilon\) from a standard normal distribution, i.e., \(\epsilon\sim\mathcal{N}(0,I_{d})\).  
3. Transform the noise by scaling it with the obtained variance, and adding it to the mean, as \(\mathbf{z}=\mu(\mathbf{h})+\sigma(\mathbf{h})\cdot\epsilon\).  

We further introduce a **Causality Preserving Module (CPM)** that takes \([\mathbf{z}\oplus x_{n,t}^{1}]\) as input and outputs estimated next step's events of a specific type depending on the input event type. For example, when the input is medication, the corresponding event type that CPM learns for the next step is the adverse event (orange arrows in Fig. 2) and vice versa (blue arrows in Fig. 2). The decoder \( p_{\theta}(x|z) \) accepts the latent embedding \( z \) and then generates \( \hat{x}_{n,t}^u \). We repeat this process for all \( t \in \{1, \ldots, T\} \) and for all \( u \in \{1, \ldots, U\} \) except for the treatment event, which yields a personalized digital twin \( \hat{X}_{n,1:T} = \{\hat{x}_{n,t}\}_{t=1}^{T} \) that closely resemble the real record \( X_{n,1:T} \).

**Counterfactual Generation.** This task aims to simulate patient trajectories under a different treatment schedule. For example, switching a patient from the treatment arm (\( \mathcal{T} \)) to the control arm (\( C \)). This task not only augments patient records but also provides a route for estimating individualized treatment effects. Additionally, it helps balance the trial data for predictive modeling and significantly reduces the sample size required for recruiting control arm participants.

Counterfactual generation starts from searching \( \tilde{X}_{k,1:T_k} \) that is the most similar patient record to \( X_{n,1:T_n} \), where \( n \in \mathcal{T} \) and \( k \in C \). Using the baseline feature of the patient \( n \), the model then generates a synthetic trajectory based on the record of patient \( k \). In this way, the generated counterfactual digital twin \( \hat{X}_{n,1:T_n} \) entails both the personal characteristics of patient \( n \) and the temporal pattern of patient \( k \).

### 3.4 Training  

Our training loss consists of two parts, as described below:  

**Generative Loss.** The loss function for the VAE consists of reconstruction loss and KL divergence. Minimizing this loss is equivalent to maximizing the Evidence Lower Bound. We use binary cross entropy as the reconstruction loss.  

\[
\mathcal{L}_1 = -\sum_{j=1}^{l_u} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \left(x_{n,t}^{u,j} \log(\hat{x}_{n,t}^{u,j}) + (1 - x_{n,t}^{u,j}) \log(1 - \hat{x}_{n,t}^{u,j})\right) + \sum_{n=1}^{N} \sum_{t=1}^{T_n} D_{KL}(q_{\phi}(z|h)||p_{\theta}(z)).
\]  

Here, \( T_n \) is the number of visits of patient \( n \), \( \hat{x}_{n,t}^{u} \) is the reconstructed version of \( x_{n,t}^{u} \) we get from the decoder of the VAE. \( p_{\theta}(z) \) is the prior distribution \( \mathcal{N}(z|0, I_d) \). \( l_u \) is the dimension of the multihot vector \( x_{n,t}^{u} \).  

**Causality-preserving Loss.** The hidden representation \( z \) of event type \( u \) is concatenated with treatment \( x_{n,t}^{1} \) to predict the next timestep \( (t+1)^{\text{th}} \) event type \( u' \neq u \). For example, the predictor in Fig. 2 is a neural network that estimates the next step’s adverse events with the input medication representation (orange arrows). Formally, the causality-preserving loss is  

\[
\mathcal{L}_2 = -\sum_{j=1}^{l_u'} \sum_{n=1}^{N} \sum_{t=1}^{T_n-1} \left(x_{n,t+1}^{u',j} \log(\hat{x}_{n,t+1}^{u',j}) + (1 - x_{n,t+1}^{u',j}) \log(1 - \hat{x}_{n,t+1}^{u',j})\right),
\]  

where \( \hat{x}_{n,t+1}^{u'} \) is predicted next step events of type \( u' \) which has a causal relationship with the input event type of TWIN. \( l_u' \) is the dimension of the multihot vector \( x_{n,t+1}^{u'} \). The reason why we include \( \mathcal{L}_2 \) is to urge TWIN to learn to generate a variant of input events that are aligned with the events in the next step satisfying the temporal causal relations, hence yielding synthetic visits with high fidelity.  

The final objective function of TWIN is

\[\mathcal{L}=\beta\mathcal{L}_{1}+\gamma\mathcal{L}_{2}.\] (9)

Here, \(\beta\) and \(\gamma\) are non-negative hyperparameters to assign importance to a specific type of loss. Reconstructed events from TWIN are combined to get a synthetic visit (Fig. 2).

### 3.5 Fidelity Evaluation

We evaluate the fidelity of the generated digital twins by considering two aspects, for the personalized generation and the counterfactual generation tasks, respectively.

**Dimension-wise probability.** This refers to the Bernoulli success probability of each dimension/feature (medication or adverse event) in the dataset. The dimension-wise probability (\(DP\)) or marginal probability is computed using the following formula:

\[DP=\frac{\text{# of visits containing the feature}}{\text{total \# of visits}}.\] (10)

If \(DPs\) of real dataset are close to \(DPs\) of synthetic data, the quality or fidelity of the synthetic data is high. To summarize the fidelity as a score, we plot real vs. synthetic \(DPs\) for features and calculate the Pearson correlation coefficient, \(r\in[-1,1]\). If \(r=1\), it is the perfect score.

**Counterfactual digital twin evaluation.** For a participant \(n\in\mathcal{T}\), we search for the first nearest neighbor \(n^{\prime}\) who is in the opposite arm \(n^{\prime}\in C\). Since we never have access to this counterfactual outcome when the target participant \(n\) was assigned to \(C\), we take participant \(n^{\prime}\) as the surrogate for the ground truth, which is a common practice in the causal inference literature [27]. When there are more than two arms: for each arm, we compare the generated record and the surrogate ground truth. We evaluate the similarity between the generated digital twin and the participant \(n^{\prime}\) by computing the personalized Pearson correlation coefficients \(r\in[-1,1]\). To summarize the fidelity as a score, we plot \(n^{\prime}\)'s \(DPs\) vs. the synthetic record's \(DPs\) for all features and calculate \(r\). We also evaluate the generated digital twin by the results of a downstream task of severe outcome prediction. We train an LSTM model with real data. We evaluate a counterfactual digital twin of \(n\) by considering their ground truth labels to be the same as \(n^{\prime}\)'s. Such that, the higher the AUROC score, the better the fidelity of the counterfactual twins.

### 3.6 Utility Evaluation

To evaluate the utility of TWIN, we observe if the additional synthetic data generated by TWIN can improve the performances for predictive tasks. To be specific, the following predictive tasks are involved in our experiments:

* ***Severe outcome prediction**: We define the occurrence of severe outcomes as the target label, which includes death and other severe adverse events. We train another LSTM predictor to make the _patient-level outcome predictions_. This model takes in all the visits of a patient as input to predict severe outcomes. This task is designed to compare the utility between TWIN and other baselines. We also check if increasing the amount of synthetic data generated by TWIN can increase the predictive performance of the downstream predictor.

* ***Adverse event prediction**: This is a _visit-wise prediction task_ designed to see how well the generated synthetic data have maintained the first causal relation. We train an MLP predictor on the data and evaluate its prediction performance. The better the performance, the better fidelity of the synthetic data is in terms of the causal-preserving quality.

* ***Data augmentation for minor groups**: ML models often struggle with imbalanced training data. To address this issue, synthetic data can be generated for classes with fewer samples using TWIN, which creates personalized data points. We evaluate how much improvement led by TWIN serving the minor group.

### 3.7 Privacy evaluation

It is critical to evaluate the privacy of synthetic data to ensure that the information disclosed in the synthetic data is not sensitive and cannot be traced back to the original data. We evaluate three types of privacy risks in the following.

**Presence disclosure.** Presence disclosure indicates that an attacker learns that a synthetic data generation model was trained using a dataset that contains patient \(n\)'s record \(X_{n:1:\mathcal{I}_{n}}\)[21]. Presence disclosure for TWIN occurs when an attacker who already has the full records for patients in set \(X\), wants to check the synthetic patient records to investigate if any members of \(X\) are actually in the training set. This attack has been referred to as a _membership inference attack_ for ML models [28]. We use sensitivity as an evaluation metric for this task. The higher the sensitivity, the easier it is for the attacker to confirm the presence of known samples in the training data. We need synthetic data and compromised evaluation data (a set of patients whose records are known by the attacker) for this task. We calculate sensitivity, \(S\), as follows:

\[S=\frac{\#\text{ of known records discovered from synthetic data}}{\text{ total }\# \text{ of known records}}.\] (11)

**Attribute disclosure.** When a data intruder may infer additional attributes about a patient, such as specific medications or adverse events, based on a subset of the data they already know about the patient, this attack is known as an attribute disclosure attack [20]. We use mean sensitivity as the evaluation metric for attribute disclosure attacks. We need synthetic data and compromised evaluation data (the set of patients whose records are partially known by the attacker) for this task. Mean sensitivity, \(MS\), is as follows:

\[MS=\frac{1}{N}\sum_{v=1}^{N}\frac{\#\text{ of unknown features of }v\text{ discovered}}{\text{ total }\#\text{ of unknown features of }v},\] (12)

where \(v\) is a compromised visit, and \(N\) is the total number of compromised visits.

**Nearest neighbor adversarial accuracy risk (NNAA).** Nearest neighbor adversarial accuracy risk is a privacy loss metric that directly measures the extent to which a generative model overfits the real dataset [37, 38]. This metric is necessary because overfitting may give rise to privacy concerns if a method is prone to fully replicating its training data when generating synthetic data. We create three datasets of the same size: real training data (\(S_{T}\)), synthetic data (\(S_{S}\)), and evaluation data (\(S_{E}\)). NNAA risk score is the difference between 1) the aggregated distance between synthetic data and evaluation data and 2) the aggregated distance between

**Table 1: Statistics of the used datasets.**

Breast Cancer Trial (NCT00174655)
| Item    | Number    |
|---|---|
| Number of patients   | 971    |
| Total number of visits | 8292    |
| Maximum number of visits per patient | 14    |
| Types of treatments | 4    |
| Types of medications | 100    |
| Types of adverse events | 56    |
| Number of patients with severe outcomes | 122    |


Lung Cancer Trial (NCT01439568) 
| Item    | Number    |
|---|---|
| Number of patients   | 77    |
| Total number of visits | 353    |
| Maximum number of visits per patient | 5    |
| Types of treatments | 3    |
| Types of medications | 100    |
| Types of adverse events | 29    |
| Number of patients with severe outcomes | 56    |


synthetic data and real data. The equation for NNAA is as follows:

\[ \text{NNAA risk score} = \text{AA}_{ES} - \text{AA}_{TS}, \tag{13} \]

where,

\[\text{AA}_{ES} = \frac{1}{2} \left( \frac{1}{N} \sum_{i=1}^{N} 1(d_{ES}(i) > d_{EE}(i)) + \frac{1}{N} \sum_{i=1}^{N} 1(d_{SE}(i) > d_{SS}(i)) \right), \tag{14}\]

\[\text{AA}_{TS} = \frac{1}{2} \left( \frac{1}{N} \sum_{i=1}^{N} 1(d_{TS}(i) > d_{TT}(i)) + \frac{1}{N} \sum_{i=1}^{N} 1(d_{ST}(i) > d_{SS}(i)) \right). \tag{15}\]

Here, \( d_{ES}(i) = \min_j \| x_E^i - x_S^j \| \) is defined as the \( L_2 \) distance between \( x_E^i \in S_E \) and its nearest neighbor in \( x_S^j \in S_S \). Similarly we can define \( d_{TS}, d_{ST} \) and \( d_{SE} \). Here, \( d_{EE}(i) = \min_{j,j \neq i} \| x_E^i - x_E^j \| \). Similarly, we can define \( d_{TT} \) and \( d_{SS} \). 1\((\cdot)\) is indicator function.




## 4. Experiments  

In this section, we present the experiment results that assess TWIN in three different aspects:  

- **Fidelity**: The degree to which the synthetic data generated by TWIN resembles real clinical trial data.  
- **Utility**: The usefulness of the synthetic data for downstream predictive tasks.  
- **Privacy**: The extent to which the synthetic data protects the privacy of individuals in real trial data.  

### 4.1 Experimental Setup  

#### 4.1.1 Data Source  
The statistics of these datasets are summarized in Table 1. The first dataset is a phase III breast cancer clinical trial (NCT00174655). There is a total of 2,887 patients who were randomly assigned to the arms to evaluate the activity of Docetaxel, given either sequentially or in combination with Doxorubicin, followed by CMF, in comparison to Doxorubicin alone or in combination with Cyclophosphamide, followed by CMF, in the adjuvant treatment of node-positive breast cancer patients. The second dataset is a Small Cell Lung Carcinoma clinical trial dataset (NCT01439568). This Phase II trial dataset contains data from both the comparator and experimental arms. 90 patients were randomly assigned to the arms to test the effect of LY2510924 and Carboplatin/Etoposide Versus Carboplatin/Etoposide in Extensive-Stage Small Cell Lung Carcinoma. We downloaded and processed the publicly available dataset from Project Data Sphere [2]. The first dataset is used to assess our algorithm's performance in terms of fidelity, utility, and privacy. The second dataset tests the model's utility performances in a very small dataset scenario.  

#### 4.1.2 Data Preprocessing  
We extracted the medications, treatments, and adverse events from the raw clinical trial data. We kept the top 100 frequent medications from each dataset. For adverse events, we selected the top 50 frequent ones. We then combined all rare events (frequency < 50) as one extra adverse event representing all rare adverse events.  

#### 4.1.3 Baseline Models  
We compare TWIN with these methods that work for synthetic EHR generation and clinical trial data generation:  

- **EVA [6]**: A generative model for generating synthetic electronic health records using conditional variational autoencoders.  
- **SynTEG [39]**: A generative model for generating synthetic electronic health record data using Wasserstein GAN.  
- **PromptEHR [31]**: A method for EHR generation with generative language models equipped with prompt learning.  
- **KNN-based method [4]**: A simple model that perturbs the real patient data by randomly extracting pieces from its nearest neighbors. It was proposed to generate tabular clinical trial data only.  

EVA, SynTEG, and PromptEHR are synthetic EHR generation models. These three models cannot abide by the constraints of specific treatment strategies for specific trial arms. Also, they are not for personalized synthetic records generation and thus do not apply to the digital twin generation. The KNN-based method was proposed for generating tabular clinical trial data. We adapt it for generating sequential records by perturbing on the visit level and then accumulating them patient-wise. Appendix A.3 contains the implementation details.  

**Table 2**: The AUROC scores for the severe outcome prediction task on the real and synthetic trial data, and a combination of both. *1xsyn* is short for synthetic data with 1× size of the corresponding real data. The best scores across a column are in bold. The performances better than the real data are underlined.  

| Model/Data       | | Breast Cancer Trial |
|------------------|--------------------|------------------|  
|                  | 1xsyn. | real+1xsyn. |   
| EVA              | 0.529 ± 0.01 | 0.760 ± 0.01 |
| SynTEG           | 0.534 ± 0.02 | 0.680 ± 0.03 |
| KNN              | 0.718 ± 0.04 | 0.750 ± 0.02 |
| PromptEHR        | 0.761 ± 0.03 | 0.777 ± 0.02 |
| TWIN             | **0.773 ± 0.01** | **0.781 ± 0.01** |
| Real Data        | 0.771 ± 0.02 | 0.648 ± 0.02 |  


| Model/Data       | | Lung Cancer Trial |  
|------------------|--------------------|------------------|  
|                  | 1xsyn. | real+1xsyn. |  
| EVA              | 0.673 ± 0.02 | 0.662 ± 0.02 |  
| SynTEG           | 0.627 ± 0.02 | 0.669 ± 0.04 |  
| KNN              | 0.673 ± 0.04 | 0.658 ± 0.02 |  
| PromptEHR        | 0.640 ± 0.03 | 0.688 ± 0.04 |  
| TWIN             | **0.733 ± 0.03** | **0.742 ± 0.03** |  
| Real Data        | 0.771 ± 0.02 | 0.648 ± 0.02 |  


## Figure 3
![Figure 3](Figure3.png) 

Figure 3: Dimension-wise probabilities for medications and adverse events (Breast Cancer Trial (NCT00174655)). The x- and  
y-axis show the dimension-wise probability for real data and synthetic data, respectively. r is Pearson correlation coefficient.  
The top row (a-e) displays the medication probability. The bottom row (f-j) displays the adverse event probability. In both cases,  
TWIN exhibit high performance with r = 0.99, and the only competitor is KNN which directly uses the raw patient data with  
small perturbation.  


## Figure 4
![Figure 4](Figure4.png) 

Figure 4: Patient-wise Pearson correlation coefficient (\( r \)) for TWIN. \( r \) is calculated by plotting \( DPs \) for each patient’s real records on the x-axis and \( DPs \) for corresponding synthetic records on the y-axis. The majority of participants have high fidelity (\( r \) close to 1).  

## Figure 5
![Figure 5](Figure5.png) 


Figure 5: Patient-wise Pearson correlation coefficient (\(r\)) for TWIN\(arm\). \(r\) is calculated by plotting \(DPs\) for each patient’s nearest neighbor on the \(x\)-axis and \(DPs\) for the corresponding counterfactual digital twin on the \(y\)-axis. The majority of participants have high fidelity (\(r\) close to 1).  


## 4.2 Fidelity

### 4.2.1. Personalized Generation

For each of the medications and adverse events, we find out their _DPs_ (Eq. (10)) separately from both real data and synthetic data generated. This works as a sanity check to ensure that the model has learned each dimension's distribution correctly. Fig. 3 shows the fidelity of data generated by TWIN and other baselines for Breast Cancer data. According to Fig. 3, TWIN shows significant benefits over EVA, SynTEG and PromptEHR, which also implies that applying EHR generation models to trial data results in suboptimal performances. On the other hand, KNN achieves a comparable performance with TWIN, which is intuitive because KNN, in essence, copies and merges the pieces from real data to create synthetic data. Nonetheless, KNN raises the privacy risk of leaking information from real data as it does not generate new data, and all the generated data can be linked back to their original patients' records. We provide the fidelity test results on the Lung Cancer trial data in Fig. 10, which induces similar observations made in the Breast Cancer data.

We also find out per patient _DPs_ to see how each synthetic record captures the corresponding real record. For each patient, we calculate the Pearson Correlation Coefficient, \(r\), between the _DPs_ of the real data and of the corresponding synthetic data. Fig. 4 shows the histogram of these Pearson Correlation Coefficients. We find that the majority of \(rs\) is close to 1, which implies the high feature-wise fidelity of the generated synthetic data.

### 4.2.2. Counterfactual Generation

We evaluate the counterfactual generation results using the strategy described in §3.5. In particular, we train an LSTM model for severe outcome prediction on the real data and then utilize the model to predict the counterfactual digital twins (i.e., the simulated patients assigned to the other treatment arm) generated by TWIN. Then, we compare the predictions with the surrogate ground truth outcomes. The obtained AUROC score is 0.733, which exhibits the high fidelity of the generated counterfactual digital twins.

We also show the patient-wise similarity of counterfactual digital twins and the ground truth records. For each patient, we compute a Pearson correlation coefficient by comparing \(DPs\) of real records of the nearest neighbor with \(DPs\) of synthetic records. We plot these coefficients in Fig. 5, which shows that for the majority of the patients, the counterfactual digital twins highly resemble the corresponding ground truth records, as most digital twins obtain high Pearson correlation coefficient \(r\) close to 1.

## Figure 6
![Figure 6](Figure6.png) 

Figure 6. Effect of the size of generated synthetic data for severe outcome prediction for the Breast Cancer Trial dataset (NCT00174655). Here, 0x means only real data, 1\(\times\)/2\(\times\)/3\(\times\) means real data+1\(\times\)/2\(\times\)/3\(\times\) synthetic data.


## 4.3 Utility

We evaluate the utility of the synthetic data in the following tasks.

#### 4.3.1. Severe outcome prediction.

**Breast Cancer Trial (NCT00174655)**: We train LSTM models for predicting severe outcomes (i.e. death or severe adverse event) using real data or the combination of synthetic and real data. The synthetic datasets are obtained by TWIN and baselines separately. Results are exhibited in Table 2. It can be seen that TWIN performs the best across all methods for both settings. The model trained by TWIN's synthetic data closely approximates the performance of real data. In addition, we further find that TWIN is the only method that generates favorable synthetic data, i.e., augmenting the real data with synthetic data leads to performance lift. In contrast, although the KNN method achieves comparable fidelity as TWIN, it does not generate favorable synthetic data. That is because the KNN method breaks the causal relations in the data by randomly replacing the events in the real visits with their neighbored data.

To evaluate the effectiveness of synthetic data in improving predictive tasks, we conducted experiments on the size of the augmented dataset. The results, shown in Fig. 6, indicate that as more synthetic data is added to the real data, the AUROC scores for TWIN consistently improve. We also compare the results with those of the KNN-based method and EVA. The results indicate that the two selected baselines struggle to generate favorable synthetic data, with only marginal improvement observed in the \(2\times\) case and inconsistent utility when varying data sizes.

**Lung Cancer Trial (NCT01439568)**: We report the experiment results on Lung Cancer data in Table 2 as well. Our method successfully generates synthetic data that enables the LSTM model to be useful for the prediction task. We also observe an improvement in performance when synthetic data is added to the real data. These results validate that TWIN presents a promising solution to unlock the application of deep learning predictive models in low-data trials.

#### 4.3.2. Adverse event prediction

In this task, we aim at evaluating the utility of the synthetic data on the visit-level prediction task. We train an MLP to predict the next time step's adverse events taking previous visits as the input. Then, the trained MLP is tested on real patient data. The generated synthetic data has the same number of


## Figure 7
![Figure 7](Figure7.png) 

Figure 7: Adverse Event Prediction AUROC scores. The red line shows the test AUROC score when the MLP is trained on real training data only. We can observe TWIN, and KNN generated synthetic data performs very similarly when the MLP is trained on synthetic data and tested on real test data.



## Figure 8
![Figure 8](Figure8.png) 



**Figure 8: Minority class data augmentation AUROC scores. Real data is augmented with 0x, 1x, 2x and 3x synthetic data for the minority class (bars from left to right in each plot).**  

patients as the real data. Results are reported in Fig. 7, where both the KNN-based method and TWIN perform similarly to the real data (represented by the red line), indicating that our method captures the temporal causal relations accurately.  

#### 4.3.3 Data augmentation for minority class  

The collected clinical trial data is usually imbalanced; e.g., in the Breast Cancer data, only 122 out of 971 patients had severe outcomes. In this experiment, our objective is to leverage the unique function of TWIN on generating personalized digital twins to augment the model’s prediction capability for minority-class patients. In detail, we identified the minority class (class 1: the class that has severe outcomes) for the Breast Cancer Trial dataset. We separated 50 patients from class 1 and added 50 more from class 0 as the test set. Then, we generated 1×, 2×, and 3× for the minority-class training data using TWIN, and merged them with the real data.  

Results are offered in Fig. 8a. By increasing the sample size of synthetic data, the performance improves for the severe outcome prediction task for 1× and 2× synthetic data for the minority class. Same experiments were also done for the Lung Cancer data, shown in Fig 8b (on the right). It is observed that TWIN leads to substantial improvement for the small dataset as the AUROC jumps to 0.81 when \( n = 90 \) synthetic data specifically for the minority class is added to the training data. This result implies the TWIN’s potential for broadening the cohort diversity and balancing the trial data.

## Figure 9
![Figure 9](Figure9.png) 

Figure 9: Presence disclosure sensitivity scores with a varying number of samples known by the attacker. (Lower sensitivity better)

Table 3: The privacy evaluation results in terms of attribute disclosure. Lower sensitivity scores are better.

| # of compromised feature | 5    | 10   | 15   | 20   |
|---|---|---|---|---|
| KNN    | **0.221** | 0.283 | 0.303 | 0.394 |
| TWIN    | 0.258 | **0.261** | **0.272** | **0.243** |

thus enabling a more comprehensive examination of the proposed treatment in clinical trials with low cost.

## 4.4 Privacy  

It is critical to measure privacy preservation when sharing synthetic clinical trial data. We evaluate the **presence disclosure** (also known as **membership inference**) risk, **attribute disclosure** risk, and **nearest neighbor adversarial accuracy (NNAA)** risk on the Breast Cancer Trial (NCT00174655) data. We compare TWIN's privacy risks with KNN-based method's, which is the only clinical trial synthetic data generation method among all baselines.  

#### 4.4.1. Presence disclosure  

When an attacker learns that TWIN was trained on a dataset that contains patient \(n\)'s record \(X_{n;1:T_{n}}\), presence disclosure occurs. We assume the attacker will ignore the visit orders and check if any synthetic record matches records from \(X_{n;1:T_{n}}\). We randomly select \(m\%\) training samples to be already known by the attackers. We set \(m\) as \(1\%,5\%,10\%\) and \(20\%\). If a visit from this known set matches a synthetic visit, the attacker thinks the model was trained with the patient. We use Eq. (11) to calculate sensitivity, \(S\). Fig. 9 shows that \(S\) of TWIN is much lower than the KNN-based model for all \(m\), where the KNN model reaches the maximum sensitivity of around \(50\%\) (i.e., attacker can identify \(50\%\) of the patients they know from the synthetic data). In a nutshell, TWIN is less prone to the presence disclosure attack independent of the number of samples known to the attacker. According to [38], if we divide the range of presence disclosure risk from 0 to 1 equally into three categories (i.e., low, medium, and high), TWIN is in low-risk category (risk < 0.333) and KNN-based model is in medium risk category (\(0.333<\) risk < 0.667).  

#### 4.4.2. Attribute disclosure  

We randomly select \(1\%\) training patient records to which the attacker has partial access. This means the attacker has access to only \(x\%\) of the features of these records. We vary \(x\) as 5, 10, 15, and 20 in this experiment. Having access to some features, the attacker tries to find \(k=5\) similar samples from the synthetic data and known records to infer the other features based on the compromised features by majority voting. We calculate mean sensitivity using Eq. (12). Table 3 shows the results of the attribute disclosure attack. We can also observe that the maximum sensitivity of KNN-based model and TWIN are 0.394 and 0.272, respectively. Similar to the setting of presence disclosure, if we divide the range of attribute disclosure risk into low, medium, and high risk categories, TWIN is in the low-risk category (risk < 0.333), and KNN-based model is in medium risk category (0.333 < risk < 0.667).  

#### 4.4.3. Nearest neighbor adversarial accuracy risk  

To assess whether the synthetic dataset generated by TWIN overfits the real training data, we compute the NNAA risk score according to Eq. (13). For the evaluation set (\(S_{E}\)), we select 100 patients' records which sums up to 751 visit records. The real training dataset (\(S_{T}\)) data also has 751 visit records, as required by the method. We then select corresponding visits from the synthetic dataset to form \(S_{S}\). TWIN has a score of 0.275 which is less than KNN-based model's risk score 0.300. According to [37], if the generator model overfits the training data, then NNAA score should be close to 0.5 ( 0.5 is considered high). Our method is better than KNN-based model in terms of overfitting risk.  

## 4.5 Ablation Study  

We remove the retrieval augmented generation (RAE) from TWIN and name the model as TWIN-RAE. The results from this ablation on the breast cancer dataset are shown in Table 4. The results show that TWIN performs better than TWIN-RAE in terms of AUROC scores for severe outcome prediction tasks on the breast cancer dataset.  

**Table 4: Ablation study. 1xsyn is short for synthetic data with 1x size of the corresponding real data.**

|               | 1 xsyn               | real+1xsyn          |
|---------------|----------------------|---------------------|
| TWIN-RAE      | \(0.6181 \pm 0.017\) | \(0.7778 \pm 0.014\) |
| TWIN          | \(0.7729 \pm 0.009\) | \(0.7810 \pm 0.007\) |


## 5. Conclusion

In this study, we proposed TWIN, which generates personalized clinical trial digital twins equipped with variational autoencoders. TWIN demonstrated outstanding results when rigorously evaluated in comparison to other baselines in the aspects of fidelity, utility, and privacy. Given the difficulty in obtaining clinical trial data and the small sample size, we anticipate that TWIN can provide a high-quality and safe alternative. The empirical analysis of privacy shows that there are low privacy concerns associated with TWIN regarding presence disclosure, attribute disclosure, and nearest neighbor adversarial accuracy risks. Although TWIN currently deals with mainly two main types of causalities in clinical trial settings, other in- or cross-event causal relations can be easily incorporated in TWIN. We intend to further enhance the sequential learning capability of TWIN in the future and try to incorporate more modalities like lab measurements and patient demographics.