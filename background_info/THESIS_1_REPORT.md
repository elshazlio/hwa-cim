## AUC School of Sciences and Engineering

## Department of Electronics and Communications Engineering

## ECNG 4980 \- Fall 2025

## Thesis I Report

# SRAM-based Hybrid AI Acceleration for Inference at the Edge

# Abanoub Emad \- Armia Samir \- Farah Wahib \- John Fahmy \- Mariam Mikhail \- Omar ElShazli

## Supervised by: Prof. Yehea Ismail

Table of Contents

**[Introduction	2](#heading=h.hmnkqizef3gw)**

[**Literature Review	2**](#literature-review)

[1.0 Introduction: The Imperative for Efficient Edge AI	2](#1.0-introduction:-the-imperative-for-efficient-edge-ai)

[2.0 The Compute-in-Memory (CiM) Paradigm: Architectures and Trade-offs	3](#2.0-the-compute-in-memory-\(cim\)-paradigm:-architectures-and-trade-offs)

[3.0 A Survey of Memory Technologies for CiM Architectures	3](#3.0-a-survey-of-memory-technologies-for-cim-architectures)

[3.1 Non-Volatile Memory (NVM) Approaches	3](#3.1-non-volatile-memory-\(nvm\)-approaches)

[3.2 Volatile Memory (SRAM) Approach	4](#3.2-volatile-memory-\(sram\)-approach)

[3.3 Rationale for a Hybrid SRAM-Based Investigation	4](#3.3-rationale-for-a-hybrid-sram-based-investigation)

[4.0 Proposed Foundation: Charge-Domain SRAM-CiM with C-2C Ladders	5](#4.0-proposed-foundation:-charge-domain-sram-cim-with-c-2c-ladders)

[4.1 Principles of Charge-Domain Computing	5](#4.1-principles-of-charge-domain-computing)

[4.2 The C-2C Capacitor Ladder for Linear MAC Operations	5](#4.2-the-c-2c-capacitor-ladder-for-linear-mac-operations)

[5.0 Hardware Implementation Challenges: Non-Idealities in the UMC 65nm Process	6](#5.0-hardware-implementation-challenges:-non-idealities-in-the-umc-65nm-process)

[5.1 Fundamental Impairments in Analog Charge-Domain Computing	6](#5.1-fundamental-impairments-in-analog-charge-domain-computing)

[5.2 Comparative Analysis: Intel 22nm FinFET vs. UMC 65nm	7](#5.2-comparative-analysis:-intel-22nm-finfet-vs.-umc-65nm)

[5.3 The Core Research Gap: Power Losses in Multi-Array Scaling	7](#5.3-the-core-research-gap:-power-losses-in-multi-array-scaling)

[6.0 The Nature of AI: A Hardware-Software Co-Design Solution	8](#6.0-the-nature-of-ai:-a-hardware-software-co-design-solution)

[6.1 The Principle of Hardware-Aware (HWA) Training	8](#6.1-the-principle-of-hardware-aware-\(hwa\)-training)

[6.2 The "Analog Foundation Models" Methodology	8](#6.2-the-"analog-foundation-models"-methodology)

[6.3 A Framework for Simulation and Validation	9](#6.3-a-framework-for-simulation-and-validation)

[7.0 Conclusion	9](#7.0-conclusion)

[**Thesis II	10**](#thesis-ii)

[Where we Stopped	10](#where-we-stopped)

[**Proposed Implementation	10**](#proposed-implementation)

[Unit Cell	10](#unit-cell)

[Array	10](#array)

[AI Testing and Workflow	10](#ai-testing-and-workflow)

[References	11](#references)

# 

# **Abstract**

# **Introduction**

# **Literature Review** {#literature-review}

## **1.0 Introduction: The Imperative for Efficient Edge AI** {#1.0-introduction:-the-imperative-for-efficient-edge-ai}

In the current era of artificial intelligence (AI), machine learning and neural networks have become deeply integrated into our daily lives. A wide range of AI-driven applications, from image classification and natural language processing to facial recognition, are now commonplace \[1\] \- \[5\]. Historically, these powerful capabilities have been delivered via a cloud-based paradigm, where user data is transmitted to massive data centers for processing. While effective, this model presents significant challenges, particularly regarding latency and data privacy. The round-trip time for data transfer can introduce noticeable delays, and the reliance on network connectivity makes services vulnerable to interruption. Furthermore, transmitting personal data to third-party servers creates inherent privacy risks, exposing users to potential data breaches or unauthorized surveillance \[6\].

These limitations have catalyzed a strategic shift toward performing AI inference directly on edge devices. However, this shift introduces a formidable challenge: executing complex AI models on power- and area-constrained hardware. Conventional computer architectures, based on the von Neumann model, are fundamentally ill-suited for this task. In these systems, memory and processing units are physically separate, necessitating constant data movement between them. This data transfer, known as the "von Neumann bottleneck," dominates the total energy consumption, rendering traditional architectures inefficient for battery-powered edge devices \[7\], \[8\].

To overcome this bottleneck, a transformative architectural paradigm known as Compute-in-Memory (CiM) has emerged. By merging arithmetic units directly within memory arrays, CiM architectures drastically reduce the energy and performance costs associated with data movement \[6\]. This literature review surveys the landscape of CiM technologies to situate a novel research direction: an investigation into a **hybrid SRAM-based AI accelerator** for edge inference. The selection of this research direction is the culmination of a nine-month investigation that systematically evaluated purely digital and purely analog paradigms, ultimately identifying a hybrid approach as the most promising path to address key scalability challenges. The primary research gap this work addresses is the characterization and mitigation of **power losses that occur when scaling across multiple analog arrays**. The proposed solution is a hardware-software co-design methodology, inspired by the concept of "Analog Foundation Models" \[9\], to create noise-resilient systems on a mature and well-characterized UMC 65nm process.

## 

## **2.0 The Compute-in-Memory (CiM) Paradigm: Architectures and Trade-offs** {#2.0-the-compute-in-memory-(cim)-paradigm:-architectures-and-trade-offs}

Understanding the Compute-in-Memory (CiM) paradigm is essential for developing next-generation AI hardware. CiM fundamentally redefines the relationship between computation and data storage. It is distinct from the related concept of Compute-near-Memory (CnM), where processing units are placed in close proximity to memory but remain separate entities. In a true CiM architecture, the memory and processing units become indistinguishable, enabling computation to occur *in situ* where data is stored \[8\]. This eliminates the load-store instructions and data transfers that characterize both von Neumann and CnM architectures, offering a more radical solution to the data movement bottleneck.

CiM can be broadly categorized into two dominant approaches: digital CiM and analog CiM (ACiM). Digital CiM architectures primarily reduce the physical distance of data movement, yielding performance and energy improvements. However, ACiM represents a more profound architectural shift. This shift is profound because it moves computation from the digital domain of Boolean logic to the analog domain of physics, leveraging Ohm's and Kirchhoff's Laws to achieve massively parallel matrix-vector multiplication in a single, constant-time operation \[8\], \[10\]. A voltage vector representing the input is applied to the rows of a memory array, and the resulting currents, modulated by the conductance of the memory cells (weights), are summed along the columns to produce the output vector.

Despite its immense potential, ACiM design involves a delicate balance of competing objectives. There is a fundamental trade-off between **linearity, throughput, and energy efficiency** \[6\]. Achieving high linearity in the analog MVM operation is critical for computational accuracy, but often requires more complex and power-hungry circuitry. Similarly, maximizing throughput may come at the cost of increased energy consumption. Any successful ACiM architecture must navigate these trade-offs effectively. While ACiM has demonstrated superior potential for energy and area efficiency compared to its digital counterparts \[6\], its practical implementation is inextricably linked to the choice of the underlying memory technology, which introduces its own set of non-idealities and constraints that must be addressed at the system level.

## **3.0 A Survey of Memory Technologies for CiM Architectures** {#3.0-a-survey-of-memory-technologies-for-cim-architectures}

The choice of memory element is the cornerstone of any CiM architecture, dictating the accelerator's performance, density, power consumption, and susceptibility to non-idealities. The physical properties of the memory cell directly influence the design of the entire system, from the peripheral circuits to the software training methodology. This section provides a critical survey of the primary volatile and non-volatile memory technologies being explored for CiM applications.

### **3.1 Non-Volatile Memory (NVM) Approaches** {#3.1-non-volatile-memory-(nvm)-approaches}

Non-volatile memory technologies are attractive for CiM due to their ability to retain stored weights without power, making them suitable for applications where models are updated infrequently. Haensch et al. provided an excellent summary that ought to be reviewed. Several NVM types have been investigated:

* **Resistive RAM (ReRAM):** ReRAM devices store information by changing their resistance, typically through the formation and rupture of a conductive filament within an insulating material \[8\]. They offer the potential for high density and analog (multi-level) state storage. However, ReRAM suffers from significant drawbacks, including high write variability due to the stochastic nature of filament formation and susceptibility to IR drops (parasitic voltage drops) in large crossbar arrays, which can degrade accuracy \[8\].  
* **Phase Change Memory (PCM):** PCM utilizes materials like Ge₂Sb₂Te₅ that can be switched between amorphous (high-resistance) and crystalline (low-resistance) states. Experimental analog AI chips have successfully employed PCM for weight storage \[7\], \[8\]. Yet, PCM faces challenges such as conductance drift over time and requires sophisticated hardware-aware training algorithms to compensate for its non-ideal characteristics and maintain acceptable accuracy \[7\], \[8\]. For instance, the IBM speech recognition chip required such co-design to counteract conductance drift in its PCM devices \[7\].  
* **Flash/SONOS:** As the most mature NVM technology, floating-gate memory like Flash and SONOS (Silicon-Oxide-Nitride-Oxide-Silicon) can store analog weights by precisely controlling the amount of charge on the floating gate. Operating these devices in the subthreshold regime is particularly advantageous, as it enables the implementation of low-conductance states with low error. This property aligns well with the typical weight distribution of neural networks, which is heavily skewed toward near-zero values, leading to high intrinsic accuracy \[11\], \[12\].  
* **Other NVMs:** Other candidates, such as Magnetoresistive RAM (MRAM), have been considered. While MRAM offers excellent endurance, its use in analog CiM is limited by drawbacks like low device resistance, which exacerbates IR drop issues, and its inherent single-bit storage capacity, which complicates the implementation of multi-bit weights \[8\].

### **3.2 Volatile Memory (SRAM) Approach** {#3.2-volatile-memory-(sram)-approach}

Static Random Access Memory (SRAM) presents a compelling alternative to NVMs for CiM accelerators. While it has well-known limitations, including lower storage density and volatility (requiring constant power to retain data) \[8\], its advantages are significant for architectural research and development.

SRAM is the workhorse memory of modern digital logic, making it ubiquitous and fully characterized in any standard CMOS process. This maturity provides unparalleled benefits: high speed, virtually unlimited endurance, and highly predictable device characteristics. These qualities make SRAM an ideal testbed for innovating on CiM architectures, as evidenced by the development of high-precision SRAM-based CiM macros capable of performing 8-bit multiply-accumulate (MAC) operations \[6\], \[13\], \[14\].

### **3.3 Rationale for a Hybrid SRAM-Based Investigation** {#3.3-rationale-for-a-hybrid-sram-based-investigation}

This research strategically selects an SRAM-based architecture as its foundation. By leveraging a mature and well-understood technology, the investigation can de-risk its primary objectives and isolate the core research questions from the material science challenges inherent in emerging NVMs. Instead of contending with device-level issues like drift, variability, and retention, the research can focus squarely on the central challenges of **hybrid architectural design, inter-array scaling, and hardware-software co-design**. This choice allows for a focused exploration of how to build robust, scalable, and noise-resilient ACiM systems using a predictable and reliable platform.

## **4.0 Proposed Foundation: Charge-Domain SRAM-CiM with C-2C Ladders** {#4.0-proposed-foundation:-charge-domain-sram-cim-with-c-2c-ladders}

Having established the rationale for an SRAM-based approach, this section details the specific analog computing technique chosen for the proposed hybrid accelerator. This technique, based on charge-domain computing, was selected as a direct response to the critical challenges of linearity and scalability that pervade the ACiM landscape.

### **4.1 Principles of Charge-Domain Computing** {#4.1-principles-of-charge-domain-computing}

In contrast to the current-domain approach common in resistive NVM crossbars, charge-domain computing performs MVM operations by manipulating and accumulating charge on capacitors. This method offers several key advantages for multi-array systems:

* **Higher Energy Efficiency:** Capacitive coupling in the charge domain has the potential to be more energy-efficient than current-mode computation, which often requires static power for biasing circuits.  
* **Passive Accumulation:** Results from multiple MAC operations can be accumulated passively through charge sharing between capacitors, eliminating the need for dedicated, power-consuming accumulation circuits.  
* **Reduced Analog Routing:** By employing local data converters (e.g., ADCs per column), the need to route sensitive analog signals over long distances is minimized, reducing susceptibility to noise and signal degradation.

### **4.2 The C-2C Capacitor Ladder for Linear MAC Operations** {#4.2-the-c-2c-capacitor-ladder-for-linear-mac-operations}

To implement the MAC operation with high linearity, this research adopts the C-2C capacitor ladder scheme proposed by Wang et al. \[6\]. This structure consists of a series of cascaded capacitor branches. Unlike a simple binary-weighted ladder where capacitor sizes grow exponentially, the C-2C ladder uses a repeating pattern: each branch contains a unit capacitor (C), and a serial capacitor of size 2C is inserted between adjacent branches.

This arrangement ensures that the contribution of each branch is binary-weighted as the signal propagates along the ladder. The output voltage VOUT is a linear superposition of the contributions from each branch, controlled by digital bits stored in the associated SRAM cells. For an illustrative 4-bit implementation as shown in the source, this linear relationship is described as \[6\]:

VOUT \= VREFi=03bi\*2i-4

Here, VREF represents the analog input voltage applied to the ladder, Bit(i) is the binary value (0 or 1\) stored in the *i*\-th SRAM cell controlling the corresponding switch, and the 2^(i-4) term represents the binary weighting achieved by the C-2C structure's passive charge division. This structure is highly scalable and area-efficient. For an 8-bit MAC unit, a C-2C ladder requires significantly fewer unit capacitors than a traditional binary-weighted capacitor ladder, making it a robust choice for this research \[6\]. While theoretically sound, the practical implementation of this architecture is subject to hardware impairments, which are magnified by the non-idealities of the chosen manufacturing process.

## **5.0 Hardware Implementation Challenges: Non-Idealities in the UMC 65nm Process** {#5.0-hardware-implementation-challenges:-non-idealities-in-the-umc-65nm-process}

A purely theoretical analysis of an ACiM architecture is insufficient; a deep understanding of hardware non-idealities is paramount. For this thesis, these real-world imperfections are not merely obstacles to be overcome but are central to the novel research angle. The non-idealities of the UMC 65nm process amplify the very phenomena under investigation \- inter-array communication losses \- providing a clear and measurable signal for analysis.

### **5.1 Fundamental Impairments in Analog Charge-Domain Computing** {#5.1-fundamental-impairments-in-analog-charge-domain-computing}

The accuracy of the C-2C capacitor ladder is susceptible to several fundamental hardware impairments, as identified by Wang et al. \[6\]:

1. **Parasitic Capacitance:** In a real layout, unintended parasitic capacitances to the substrate and between metal layers disrupt the perfect 1:2 capacitance ratio required by the C-2C ladder. This deviation introduces non-linearity into the MAC operation.  
2. **Capacitor Mismatch:** Due to manufacturing variations, the actual capacitance values will deviate from their intended design. To maintain accuracy, this deviation should be less than one-third of a least significant bit (LSB). For an 8-bit system with a 1V dynamic range, this translates to a deviation of less than 1.3 mV \[6\].  
3. **Switched-Capacitor Noise (kBT/C):** The thermal noise associated with charging and discharging capacitors (kBT/C noise) sets a fundamental limit on precision. To ensure this noise is comparable to or less than quantization noise in an 8-bit system, a minimum capacitance of 3.3 fF is required \[6\].

### 

### **5.2 Comparative Analysis: Intel 22nm FinFET vs. UMC 65nm** {#5.2-comparative-analysis:-intel-22nm-finfet-vs.-umc-65nm}

The target UMC 65nm process presents a different set of challenges and opportunities compared to the advanced Intel 22nm FinFET process used in the foundational C-2C paper. The following table summarizes these differences:

| Parameter | Intel 22nm FinFET | Target UMC 65nm |
| :---- | :---- | :---- |
| **Unit capacitance (8-bit)** | 2 fF | 15-20 fF (estimated) |
| **Mismatch @ 2fF** | 1.4 mV | \~5-7 mV (worse) |
| **kBT/C noise requirement** | 3.3 fF | 3.3 fF (same physics) |
| **Parasitic C\_p/C** | \~5-10% | \~20-50% |
| **Serial cap compensation** | 2.1-2.2C | 2.3-2.7C (more needed) |
| **Total C-2C area (8-bit)** | Very small | 5-10× larger |

This comparison reveals several critical implications. The 65nm process is expected to exhibit significantly higher parasitic capacitance, worse device mismatch, and higher transistor leakage currents. To meet mismatch and noise requirements, the unit capacitors will need to be much larger, leading to a 5-10× increase in area. While these characteristics might be seen as disadvantages for a commercial product, they are highly beneficial for this research. The larger parasitics will amplify the inter-array communication losses, creating a clearer, more easily measurable signal for studying the effects of charge-sharing efficiency and for developing effective compensation techniques.

### **5.3 The Core Research Gap: Power Losses in Multi-Array Scaling** {#5.3-the-core-research-gap:-power-losses-in-multi-array-scaling}

This thesis aims to address a critical, well-defined research gap: the characterization and mitigation of **power losses when stringing multiple analog arrays together**. As AI models grow, they must be mapped across numerous CiM arrays. The method of communication between these arrays is a dominant factor in overall system efficiency. Conventional methods each have significant drawbacks:

1. **Digital Conversion (ADC/DAC):** Converting the analog output of one array to digital, routing it, and then converting it back to analog for the next array incurs a high cost in both power and latency.  
2. **Analog Voltage-Mode Buffers:** Using active buffers to propagate analog voltage signals between arrays suffers from cumulative voltage drops and noise accumulation.  
3. **Analog Current-Mode Communication:** Propagating signals in the current domain is prone to IR drops along interconnects and is sensitive to device mismatch.

Given the significant power, latency, and noise penalties of these conventional digital and active-analog methods, this research posits a fundamentally different approach. This leads to the **novel research question** that forms the core of this thesis: **Can the principle of passive, capacitive charge-domain coupling be extended to handle inter-array communication, and what are the associated scaling effects and required compensation techniques, particularly within a process technology like UMC 65nm that is characterized by high parasitics?** This question establishes a unique and achievable research contribution.

## **6.0 The Nature of AI: A Hardware-Software Co-Design Solution** {#6.0-the-nature-of-ai:-a-hardware-software-co-design-solution}

Creating a functional AI system from imperfect analog hardware requires a holistic approach where hardware design and software training are deeply intertwined. Hardware-software co-design is not an afterthought but an essential and integrated methodology for achieving high performance and accuracy. This section outlines the specific "Analog Foundation Models" approach that will be used to validate the thesis by training neural network models that are inherently resilient to the characterized physical limitations of the designed hardware.

### **6.1 The Principle of Hardware-Aware (HWA) Training** {#6.1-the-principle-of-hardware-aware-(hwa)-training}

The central tenet of hardware-aware (HWA) training is to make the software model cognizant of the physical imperfections of the hardware during its training phase. As illustrated in the co-design flow proposed by Haensch et al., information about material properties, circuit non-idealities, noise, and variability must be fed back into the software training loop \[8\]. By exposing the model to these hardware-specific errors during training, the final software model learns to be robust against them, effectively compensating for the physical limitations of the analog substrate and achieving high system-level accuracy.

### **6.2 The "Analog Foundation Models" Methodology** {#6.2-the-"analog-foundation-models"-methodology}

This thesis will adopt the specific HWA training methodology outlined in "Analog Foundation Models" \[9\], which provides a scalable and efficient framework for creating noise-resilient models. The process involves three distinct steps:

1. **Hardware Noise Profiling:** The initial step requires establishing a high-fidelity hardware noise model. This will be achieved by extracting statistical error profiles from extensive transient and Monte Carlo simulations of the designed 8x8 or 16x16 C-2C arrays within the Cadence Virtuoso environment. These simulations will capture the effects of capacitor mismatch, parasitics, and leakage in the UMC 65nm process, yielding a precise noise profile for the target hardware.  
2. **Noise-Aware Training Loop:** The extracted hardware noise profiles are then injected directly into a software training loop, implemented in PyTorch. As a neural network model (e.g., a transformer block) is trained, its forward and backward passes are perturbed by the simulated hardware noise. This forces the model's weights to converge to a solution that is inherently robust to the specific error characteristics of the 65nm C-2C architecture.  
3. **Data Distillation:** To make the training process feasible for academic research without requiring massive, proprietary pre-training datasets, this methodology employs data distillation. A larger, pre-trained "teacher" model is used to generate synthetic data, which then serves as the training set for the smaller, noise-aware "student" model. This approach effectively transfers the knowledge from the large model while enabling robust HWA training.

### **6.3 A Framework for Simulation and Validation** {#6.3-a-framework-for-simulation-and-validation}

This methodology translates into a concrete and achievable validation plan for the thesis, centered around an end-to-end simulation testbench. The workflow is as follows:

* **Pre-processing (Python/MATLAB):** Small but realistic neural network layers, such as a micro-MLP for MNIST classification, will be used to generate input vectors and expected output patterns.  
* **Simulation (Cadence ADE):** These test patterns will be converted into stimuli using VerilogA behavioral models and applied to the detailed 65nm circuit schematic in the Cadence Analog Design Environment. Transient simulations will be run to capture the circuit's analog behavior.  
* **Post-processing (Python):** The analog output data from the simulation will be extracted and compared against a golden reference to measure computational accuracy. Critically, this end-to-end framework enables the central quantitative goal of the thesis: to perform a detailed power breakdown analysis that precisely measures and compares the energy consumed by core computation, peripheral data conversion, and \- most importantly \- the inter-array communication link.

This systematic simulation framework provides a robust and practical path for executing the novel research contribution of the thesis and validating its central hypotheses.

## **7.0 Conclusion** {#7.0-conclusion}

This literature review has established the context and rationale for a novel investigation into hybrid SRAM-based AI accelerators. The analysis began by highlighting the overarching challenge: the need for power-efficient AI inference at the edge and the fundamental limitations of the von Neumann architecture. The constant movement of data in conventional systems is a major source of energy consumption, making them unsuitable for power-constrained edge devices.

Analog Compute-in-Memory (ACiM) was presented as a promising paradigm to overcome this bottleneck by performing computations directly within memory arrays. A survey of the underlying memory technologies revealed a critical trade-off between emerging Non-Volatile Memories (NVMs), which offer high density but suffer from material science challenges, and mature SRAM, which provides a predictable and robust platform ideal for architectural innovation. This led to the strategic decision to focus on an SRAM-based system to isolate architectural questions from device physics.

The review then identified a specific, high-impact research gap: characterizing and mitigating the power losses that occur during **inter-array communication** in scaled ACiM systems. This problem, which is critical for mapping large AI models onto hardware, is amplified and made more measurable by the non-idealities inherent in mature process nodes like UMC 65nm. This research proposes a holistic contribution: the design of a novel, charge-domain SRAM architecture to probe inter-array scaling losses, coupled with a rigorous hardware-software co-design methodology that validates the system's resilience, thereby paving a viable path toward scalable and efficient AI accelerators for the edge.

# Thesis II {#thesis-ii}

## Where we Stopped {#where-we-stopped}

To date, we have successfully implemented and validated a charge-domain multiply-accumulate (MAC) unit cell in UMC 65nm technology, integrating a standard 6T SRAM bitcell with a passive C-2C capacitor ladder branch. Our verification phase involved detailed transient simulations in Cadence Virtuoso, where we proved the fundamental physics of 1-bit multiplication by correlating stored SRAM weights with analog input activation pulses to produce distinct, predictable charge-redistribution levels at the output node. We resolved critical design bottlenecks, specifically the capacitive sizing mismatch between the vertical shunt (51.16 fF) and series capacitors (101.74 fF), and successfully emulated array loading effects to stabilize the analog output. Furthermore, we established a precise control sequence using pulsed precharge and word-line timing, ensuring the charge-domain computation settles correctly within the target operating cycle. Moving forward, we will transition from individual bitcell validation to the design of an integrated SRAM macro where multiple unit cells are arrayed to demonstrate multi-bit accumulation through passive charge sharing.

## Proposed Implementation {#proposed-implementation}

### Unit Cell {#unit-cell}

The design will transition from the validated 1-bit proof-of-concept to a full **8-bit C-2C capacitor ladder** to achieve high computational precision. Utilizing the UMC 65nm process, we will implement **MIM capacitors** configured in a series-parallel arrangement to maintain the critical **1:2 ratio** required for binary weighting. To mitigate the process-specific mismatch and high parasitic capacitance (\~40%) identified during the 1-bit validation, the **serial 2C capacitors will be iteratively tuned (oversized to \~2.5C)** in the layout to ensure linear charge redistribution across all 256 input codes.

### Array {#array}

The unit cells will be integrated into a **scalable 8×8 (or 16×16) macro**, providing a manageable testbed to characterize multi-tile scaling. Unlike traditional digital ALUs, this array executes accumulation through **passive charge sharing**. By connecting the outputs of multiple C-2C ladders to a single shared column node, the multiplication results from each row are summed and averaged "for free" via basic physics, eliminating the energy overhead of active current-mode circuitry or digital adder trees. To complete the macro, **row-wise R-2R DACs** will be implemented to provide stable input activation voltages, and **column-wise SAR-ADCs** will digitize the accumulated results.

### AI Testing and Workflow {#ai-testing-and-workflow}

Validation will follow an **"Analog Foundation Models" methodology** to bridge the gap between imperfect analog hardware and high-level AI accuracy. The workflow consists of three stages:  
1\. **Hardware Noise Profiling:** We will extract statistical error profiles, including mismatch and leakage, through **Monte Carlo simulations** in Cadence Virtuoso to build a high-fidelity noise model of the 65nm array.  
2\. **Noise-Aware Training:** These profiles will be injected into a **PyTorch-based training loop**, forcing the weights of a target neural network (e.g., a micro-MLP for MNIST) to converge on solutions inherently robust to our hardware's non-idealities.  
3\. **End-to-End Validation:** We will use **data distillation**, where a larger "teacher" model generates synthetic data for our noise-resilient "student" model, allowing us to measure system-level accuracy and perform a detailed power breakdown of computation versus communication losses.

# Standards Involved in our Work

# Plan for Design Validation

# Feasibility and Economics

# Societal, Environmental, and Ethical Considerations

# Implementation Gantt Chart

# Conclusion

# References {#references}

\[1\] A. Krizhevsky, I. Sutsevker, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, 2012, pp. 1097–1105. 

\[2\] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2014, pp. 580–587. 

\[3\] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word representations in vector space,” in *Proc. 1st Int. Conf. Learn. Represent. (ICLR)*, 2013, pp. 1–12. 

\[4\] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf, “DeepFace: Closing the gap to human-level performance in face verification,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2014, pp. 1701–1708. 

\[5\] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2016, pp. 779–788. 

\[6\] C. Wang *et al.*, "A Charge-Domain SRAM Compute-in-Memory Macro With C-2C Ladder-Based 8-Bit MAC Unit in 22-nm FinFET Process for Edge Inference," *IEEE Journal of Solid-State Circuits*, vol. 58, no. 4, pp. 942-953, Apr. 2023\. 

\[7\] S. Ambrogio *et al.*, "An analog AI chip for speech recognition with computational memory," *Nature*, vol. 620, pp. 779–786, Aug. 2023\. 

\[8\] W. Haensch *et al.*, "Compute in‐Memory with Non‐Volatile Elements for Neural Networks: A Review from a Co-Design Perspective," *Advanced Materials*, vol. 35, no. 37, p. 2204944, Sep. 2023\. 

\[9\] M. J. Rasch *et al.*, "Analog Foundation Models," Preprint, 2025\. 

\[10\] C. Li *et al.*, "Multiply–accumulate operation in memristor crossbar arrays for analog computing," *Journal of Semiconductors*, vol. 42, no. 1, p. 013104, Jan. 2021\. 

\[11\] T. P. Xiao *et al.*, "An Accurate, Error-Tolerant, and Energy-Efficient Neural Network Inference Engine Based on SONOS Analog Memory," *IEEE Transactions on Electron Devices*, vol. 68, no. 10, pp. 4886-4894, Oct. 2021\. 

\[12\] B. Moyer, "Is In-Memory Compute Still Alive?," *Semiconductor Engineering*, Jan. 11, 2024\. \[Online\]. Available: https://semiengineering.com/is-in-memory-compute-still-alive/ 

\[13\] C. Wang, "Ultra-low Voltage Static Random Access Memory Design for Energy-Constrained Applications," Ph.D. Thesis, Nanyang Technological University, Singapore, 2015\. 

\[14\] C. Shin, "Advanced MOSFET Designs and Implications for SRAM Scaling," Ph.D. Dissertation, University of California, Berkeley, 2011\.

---

## Addendum — implementation scope (repository, 2026)

The **software testbed** (`hwa-cim`) and the **current Virtuoso schematic milestone** align on a **4×4** UMC 65 nm **SRAM–CiM** macro with integrated **decoder, DAC, and SAR ADC** (see `background_info/Bird's Eye View of Our Thesis.md`). Earlier body text that discusses **8×8 or 16×16** arrays remains valid as **literature and scaling** motivation; **first-extract / first-tapeout scope** for this project is **4×4** unless a future design review changes it. For software–hardware traceability and open follow-ups, see `docs/software_mission_followups.md`.