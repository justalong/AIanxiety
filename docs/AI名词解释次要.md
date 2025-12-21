### 技术框架与工具（Technical frameworks and tools）

| 英文简称 | 英文全称                          | 中文翻译         | 解释说明                     |
|----------|-----------------------------------|----------------|----------------------------|
| TF       | TensorFlow                        | TensorFlow框架 | 谷歌开源深度学习框架         |
| PyTorch  | PyTorch                           | PyTorch框架    | 动态图优先的深度学习框架     |
| Keras    | Keras                             | Keras接口      | 高层神经网络API库            |
| ONNX     | Open Neural Network Exchange      | 开放神经网络交换格式 | 跨框架模型互操作标准     |
| CUDA     | Compute Unified Device Architecture | CUDA架构      | NVIDIA GPU并行计算平台       |
| TPU      | Tensor Processing Unit            | 张量处理器     | 谷歌专为深度学习设计的芯片   |
| Jupyter  | Jupyter Notebook                  | Jupyter笔记本  | 交互式编程与数据分析工具     |
| Colab    | Google Colab                      | Google Colab   | 云端免费GPU计算平台          |
| Hugging Face | Hugging Face Transformers      | Hugging Face库 | NLP模型与数据集开源社区      |
| MLflow   | MLflow                            | MLflow工具     | 机器学习全生命周期管理平台   |
| LangChain | LangChain                        | LangChain框架  | 大语言模型应用开发工具链     |
| TensorBoard | TensorBoard                     | TensorBoard    | 模型训练可视化工具           |
| OpenCV   | Open Source Computer Vision Library | OpenCV库     | 开源计算机视觉库             |
| Spark MLlib | Apache Spark MLlib             | Spark MLlib    | 分布式机器学习库             |



### 生成模型 (Generative Models)

| 英文简称 | 英文全称                     | 中文含义     | 中文解释                               |
| -------- | ---------------------------- | -------- | -------------------------------------- |
| GM       | Generative Model           | 生成模型     | 学习数据分布并生成新数据的模型。       |
| GAN      | Generative Adversarial Network | 生成对抗网络 | 通过生成器和判别器的对抗学习生成数据。 |
| Generator | Generator                  | 生成器       | 在GAN中生成假数据的网络。             |
| Discriminator | Discriminator              | 判别器       | 在GAN中区分真实数据和假数据的网络。   |
| Latent Space | Latent Space               | 潜在空间     | 生成模型从中采样生成数据的低维空间。 |
| VAE      | Variational Autoencoder      | 变分自编码器 | 一种基于变分推断的生成模型。         |
| Encoder  | Encoder                      | 编码器       | 将输入数据映射到潜在空间的网络。       |
| Decoder  | Decoder                      | 解码器       | 将潜在空间向量映射回数据空间的网络。 |
| Autoencoder | Autoencoder                | 自编码器     | 学习数据压缩表示并重建数据的网络。   |
| Normalizing Flow | Normalizing Flow         | 正则化流     | 通过一系列可逆变换构建复杂的概率分布。 |
| Diffusion Model | Diffusion Model            | 扩散模型     | 通过逐步去噪生成数据。               |
| Markov Chain Monte Carlo | Markov Chain Monte Carlo (MCMC) | 马尔可夫链蒙特卡罗 | 一类用于从复杂分布中采样的算法。   |
| Likelihood | Likelihood                 | 似然         | 给定模型参数下观测到数据的概率。     |
| Prior    | Prior                        | 先验         | 模型参数的初始概率分布。             |
| Posterior | Posterior                    | 后验         | 给定数据后模型参数的概率分布。       |
| Sampling | Sampling                     | 采样         | 从概率分布中随机抽取样本。           |
| Image Generation | Image Generation         | 图像生成     | 生成新的图像数据。                 |
| Text Generation | Text Generation          | 文本生成     | 生成新的文本数据。                 |
| Audio Synthesis | Audio Synthesis          | 音频合成     | 生成新的音频数据。                 |
| Data Augmentation | Data Augmentation        | 数据增强     | 使用生成模型扩充训练数据。         |


### 专家系统 (Expert Systems)

| 英文简称 | 英文全称                     | 中文含义     | 中文解释                               |
| -------- | ---------------------------- | -------- | -------------------------------------- |
| ES       | Expert System                | 专家系统   | 模拟人类专家解决特定领域问题的计算机系统。 |
| Knowledge Base | Knowledge Base           | 知识库     | 存储领域专家知识的仓库。               |
| Inference Engine | Inference Engine         | 推理引擎   | 利用知识库中的知识进行推理的程序。     |
| Explanation Facility | Explanation Facility   | 解释机制   | 向用户解释系统推理过程的功能。         |
| Domain Expert | Domain Expert            | 领域专家   | 在特定领域拥有专业知识的人员。         |
| Heuristic Rule | Heuristic Rule           | 启发式规则 | 基于经验的解决问题规则。             |
| Production Rule | Production Rule          | 产生式规则 | “如果-那么”形式的知识表示。         |
| Conflict Resolution | Conflict Resolution      | 冲突解决   | 当多个规则可以应用时选择哪个。       |
| Case-Based Reasoning | Case-Based Reasoning (CBR) | 基于案例推理 | 通过借鉴以往案例解决新问题。         |
| Hybrid System | Hybrid System            | 混合系统   | 结合多种AI技术的系统。               |

### 知识表示与推理 (Knowledge Representation and Reasoning)

| 英文简称 | 英文全称                     | 中文含义         | 中文解释                               |
| -------- | ---------------------------- | ------------ | -------------------------------------- |
| KR       | Knowledge Representation     | 知识表示       | 如何在计算机中表达和存储知识。         |
| KR&R     | Knowledge Representation and Reasoning | 知识表示与推理 | 研究如何表示知识并进行逻辑推导。       |
| Ontology | Ontology                     | 知识本体       | 对特定领域概念及其关系的明确规范。     |
| Semantic Network | Semantic Network         | 语义网络       | 使用节点和边表示概念及其关系的图结构。 |
| Frame    | Frame                        | 框架         | 一种结构化的知识表示方法。           |
| Rule-Based System | Rule-Based System        | 基于规则系统   | 使用规则进行推理的系统。               |
| Inference Engine | Inference Engine         | 推理引擎       | 执行推理过程的程序。                 |
| Forward Chaining | Forward Chaining         | 前向推理       | 从已知事实推导新事实。               |
| Backward Chaining | Backward Chaining        | 反向推理       | 从目标反向寻找支持证据。             |
| Logic Programming | Logic Programming        | 逻辑编程       | 使用逻辑语句进行编程的范式。         |
| First-Order Logic | First-Order Logic (FOL)  | 一阶逻辑       | 一种强大的形式逻辑系统。             |
| Propositional Logic | Propositional Logic      | 命题逻辑       | 一种基本的形式逻辑系统。             |
| Knowledge Graph  | Knowledge Graph          | 知识图谱       | 由实体和关系组成的图状知识库。       |
| RDF      | Resource Description Framework | 资源描述框架   | 一种用于描述网络资源的W3C标准。       |
| SPARQL   | SPARQL Protocol and RDF Query Language | SPARQL       | 查询RDF数据的语言。                  |
| Reasoning  | Reasoning                    | 推理         | 从已知信息得出结论的过程。           |
| Abduction  | Abduction                    | 溯因推理       | 从观察到的结果推断最可能的解释。     |
| Induction  | Induction                    | 归纳推理       | 从个别案例推广到一般规律。           |
| Deduction  | Deduction                    | 演绎推理       | 从一般规律推导出个别结论。           |
| Common Sense Reasoning | Common Sense Reasoning   | 常识推理       | 模拟人类日常的推理能力。             |


### 智能控制 (Intelligent Control)

| 英文简称 | 英文全称                     | 中文含义     | 中文解释                               |
| -------- | ---------------------------- | -------- | -------------------------------------- |
| IC       | Intelligent Control          | 智能控制   | 使用AI技术设计和实现控制系统。         |
| Control System | Control System           | 控制系统   | 管理和调节系统行为的系统。           |
| Feedback Control | Feedback Control         | 反馈控制   | 基于系统输出调整控制输入的机制。       |
| Feedforward Control | Feedforward Control      | 前馈控制   | 基于系统输入预测并调整控制输入的机制。 |
| Adaptive Control | Adaptive Control         | 自适应控制 | 能根据系统变化调整自身参数的控制。     |
| Robust Control | Robust Control           | 鲁棒控制   | 在存在不确定性时仍能保持性能的控制。   |
| Optimal Control | Optimal Control          | 最优控制   | 寻找使系统性能指标最优的控制策略。   |
| Learning Control | Learning Control         | 学习控制   | 通过经验学习改善控制性能。           |
| Fuzzy Control | Fuzzy Control            | 模糊控制   | 基于模糊逻辑的控制方法。             |
| Neural Network Control | Neural Network Control   | 神经网络控制 | 使用神经网络实现控制。               |
| Evolutionary Control | Evolutionary Control     | 进化控制   | 使用进化计算优化控制策略。         |
| Hybrid Control | Hybrid Control           | 混合控制   | 结合多种控制方法的控制。             |
| System Identification | System Identification    | 系统辨识   | 建立系统动态模型的過程。             |
| State Space Representation | State Space Representation | 状态空间表示 | 用状态变量描述系统动态行为。         |
| Controllability | Controllability          | 可控性       | 系统状态能否通过控制输入达到任意值。 |
| Observability | Observability            | 可观性       | 系统状态能否通过输出测量值确定。     |
| Actuator | Actuator                     | 执行器       | 将控制信号转化为物理作用的装置。     |
| Sensor   | Sensor                       | 传感器       | 测量系统状态和输出的装置。           |
| Stability | Stability                    | 稳定性       | 系统在受到扰动后恢复平衡的能力。     |
| Performance | Performance                | 性能         | 控制系统满足设计要求的程度。         |


### 博弈论与多智能体决策 (Game Theory and Multi-Agent Decision Making)

| 英文简称 | 英文全称                     | 中文含义           | 中文解释                               |
| -------- | ---------------------------- | -------------- | -------------------------------------- |
| GT       | Game Theory                  | 博弈论             | 研究策略互动中理性决策的数学理论。     |
| MADM     | Multi-Agent Decision Making  | 多智能体决策       | 多个智能体如何做出联合决策。         |
| Agent    | Agent                        | 智能体           | 参与博弈或决策的独立实体。           |
| Strategy | Strategy                     | 策略             | 智能体在博弈中选择的行动方案。       |
| Payoff   | Payoff                       | 收益/回报        | 智能体采取行动后获得的结果或奖励。     |
| Utility  | Utility                      | 效用             | 智能体对不同结果的偏好程度。         |
| Game     | Game                         | 博弈             | 智能体之间策略互动的场景。           |
| Zero-Sum Game | Zero-Sum Game              | 零和博弈         | 一个参与者的收益等于其他参与者的损失。 |
| Non-Zero-Sum Game | Non-Zero-Sum Game        | 非零和博弈       | 参与者的总收益不恒定的博弈。         |
| Nash Equilibrium | Nash Equilibrium         | 纳什均衡         | 所有参与者都不愿单方面改变策略的状态。 |
| Pareto Optimality | Pareto Optimality        | 帕累托最优       | 没有其他结果能使至少一个参与者更好，且不使任何参与者更差。 |
| Mechanism Design | Mechanism Design         | 机制设计         | 设计规则以激励理性智能体达到期望结果。 |
| Auction  | Auction                      | 拍卖             | 一种资源分配的博弈机制。             |
| Bargaining | Bargaining                   | 议价             | 多个智能体协商达成协议的过程。       |
| Coalition | Coalition                    | 联盟             | 一组为共同目标合作的智能体。         |
| Voting   | Voting                       | 投票             | 一种集体决策机制。                   |
| Reputation | Reputation                   | 声誉             | 智能体基于过去行为的评价。           |
| Trust    | Trust                        | 信任             | 一个智能体对另一个智能体行为的预期。   |
| Cooperation | Cooperation                | 合作             | 多个智能体共同行动以实现共同利益。   |
| Coordination | Coordination               | 协调             | 管理多个智能体的行动以避免冲突。     |


### 时间序列分析 (Time Series Analysis)

| 英文简称 | 英文全称                     | 中文含义       | 中文解释                               |
| -------- | ---------------------------- | ---------- | -------------------------------------- |
| TSA      | Time Series Analysis         | 时间序列分析   | 分析随时间变化的数据序列的方法。       |
| Time Series | Time Series                | 时间序列     | 按时间顺序排列的数据点序列。           |
| Trend    | Trend                        | 趋势         | 时间序列数据长期变化的模式。           |
| Seasonality | Seasonality                | 季节性       | 在固定时间间隔内重复出现的模式。       |
| Cycle    | Cycle                        | 周期性       | 比季节性更长期的波动模式。             |
| Random Noise | Random Noise               | 随机噪声     | 时间序列中无法预测的随机波动。       |
| Stationarity | Stationarity               | 平稳性       | 时间序列的统计特性不随时间变化。       |
| Autocorrelation | Autocorrelation          | 自相关       | 时间序列在不同时间点的相关性。       |
| Partial Autocorrelation | Partial Autocorrelation | 偏自相关     | 剔除中间点影响后的自相关。           |
| ARIMA    | Autoregressive Integrated Moving Average | 自回归差分滑动平均模型 | 一种常用的时间序列预测模型。         |
| AR       | Autoregressive               | 自回归       | 当前值与过去值的线性组合相关。         |
| MA       | Moving Average               | 滑动平均       | 当前值与过去误差项的线性组合相关。     |
| ACF      | Autocorrelation Function     | 自相关函数   | 度量时间序列自相关性的函数。         |
| PACF     | Partial Autocorrelation Function | 偏自相关函数 | 度量偏自相关性的函数。               |
| Decomposition | Decomposition            | 分解         | 将时间序列分解为趋势、季节性和残差。 |
| Forecasting | Forecasting                | 预测         | 基于历史数据预测未来值。             |
| Smoothing  | Smoothing                  | 平滑         | 减少时间序列中的噪声。               |
| Exponential Smoothing | Exponential Smoothing    | 指数平滑       | 对近期观测值赋予更高权重的平滑方法。 |
| Holt-Winters | Holt-Winters               | Holt-Winters模型 | 包含趋势和季节性的指数平滑方法。     |
| Unit Root Test | Unit Root Test             | 单位根检验     | 检验时间序列是否平稳。               |


### AI辅助设计 (AI-Assisted Design)

| 英文简称 | 英文全称                     | 中文含义       | 中文解释                               |
| -------- | ---------------------------- | ---------- | -------------------------------------- |
| AI-Assisted Design | AI-Assisted Design       | 人工智能辅助设计 | 使用人工智能技术辅助人类进行设计。     |
| Generative Design | Generative Design        | 创成式设计     | 使用AI算法自动生成设计方案。         |
| Design Optimization | Design Optimization      | 设计优化     | 使用AI算法改进现有设计。             |
| Parametric Design with AI | Parametric Design with AI | AI驱动的参数化设计 | 使用AI控制设计参数以探索设计空间。 |
| Computational Design | Computational Design     | 计算设计       | 使用算法和代码进行设计。             |
| AI-Powered CAD | AI-Powered CAD           | AI驱动的CAD   | 集成AI功能的计算机辅助设计软件。     |
| Design Automation | Design Automation        | 设计自动化     | 使用AI自动完成部分或全部设计过程。   |
| Design Space Exploration | Design Space Exploration | 设计空间探索   | 使用AI帮助设计师探索各种设计可能性。 |
| User-Centered AI Design | User-Centered AI Design | 以用户为中心的AI设计 | 在设计过程中考虑用户需求和反馈。   |
| AI for Conceptual Design | AI for Conceptual Design | AI用于概念设计 | 使用AI帮助设计师生成初步设计概念。 |
| AI for Material Selection | AI for Material Selection | AI用于材料选择 | 使用AI推荐合适的材料。             |
| AI for Structural Analysis | AI for Structural Analysis | AI用于结构分析 | 使用AI进行设计结构的分析和优化。   |
| AI for Architectural Design | AI for Architectural Design | AI用于建筑设计 | 使用AI辅助建筑设计和规划。         |
| AI for Industrial Design | AI for Industrial Design | AI用于工业设计 | 使用AI辅助产品设计。               |
| AI for Fashion Design | AI for Fashion Design    | AI用于时尚设计 | 使用AI辅助服装和配饰设计。         |
| AI for Graphic Design | AI for Graphic Design    | AI用于平面设计 | 使用AI辅助视觉传达设计。           |
| Human-AI Collaboration in Design | Human-AI Collaboration in Design | 人机协作设计 | 设计师与AI共同进行设计创作。       |
| Evaluation of AI-Assisted Design Tools | Evaluation of AI-Assisted Design Tools | AI辅助设计工具评估 | 衡量AI设计工具的有效性和用户体验。 |
| Ethical Considerations in AI Design | Ethical Considerations in AI Design | AI设计伦理     | 探讨在设计中使用AI的道德问题。     |
| The Future of AI in Design | The Future of AI in Design | AI在设计的未来 | 展望AI如何改变设计行业的未来。     |


### 情感计算 (Affective Computing)

| 英文简称 | 英文全称                     | 中文含义     | 中文解释                               |
| -------- | ---------------------------- | -------- | -------------------------------------- |
| Affective Computing | Affective Computing        | 情感计算     | 研究和开发能够识别、理解、模拟人类情感的系统。 |
| Emotion  | Emotion                      | 情感       | 一种复杂的心理状态，涉及生理和认知。   |
| Sentiment | Sentiment                    | 情绪       | 较为主观和短暂的情感体验。             |
| Mood     | Mood                         | 心境       | 持续时间较长、强度较低的情感状态。     |
| Facial Expression Recognition | Facial Expression Recognition (FER) | 面部表情识别 | 通过分析面部图像识别情感。             |
| Speech Emotion Recognition | Speech Emotion Recognition (SER) | 语音情感识别 | 通过分析语音信号识别情感。             |
| Text-based Emotion Recognition | Text-based Emotion Recognition | 基于文本的情感识别 | 通过分析文本内容识别情感。             |
| Physiological Signal Analysis | Physiological Signal Analysis | 生理信号分析 | 通过心率、皮肤电导等识别情感。         |
| Multimodal Emotion Recognition | Multimodal Emotion Recognition | 多模态情感识别 | 结合多种模态信息识别情感。           |
| Emotion Modeling | Emotion Modeling           | 情感建模     | 构建情感的计算模型。                 |
| Dimensional Model of Emotion | Dimensional Model of Emotion | 情感维度模型 | 用维度（如效价和唤醒度）描述情感。     |
| Categorical Model of Emotion | Categorical Model of Emotion | 情感类别模型 | 将情感划分为离散的类别（如喜怒哀乐）。 |
| Emotion Synthesis | Emotion Synthesis          | 情感合成     | 使计算机系统能够表达情感。           |
| Affective Interaction | Affective Interaction      | 情感交互     | 计算机系统与用户进行情感交流。       |
| Empathy  | Empathy                      | 共情       | 理解和分享他人情感的能力。             |
| Emotional Intelligence | Emotional Intelligence   | 情绪智力     | 识别、理解和管理自身及他人情感的能力。 |
| Human-Computer Affective Interaction | Human-Computer Affective Interaction (HCAI) | 人机情感交互 | 研究人与计算机之间的情感互动。         |
| Affective Feedback | Affective Feedback       | 情感反馈     | 计算机系统向用户提供情感相关的反馈。 |
| Emotion Regulation | Emotion Regulation       | 情绪调节     | 管理和改变情感反应的过程。           |
| Social Signal Processing | Social Signal Processing   | 社会信号处理 | 分析和理解人类社交行为中的信号。     |


### 进化计算 (Evolutionary Computation)

| 英文简称 | 英文全称                     | 中文含义         | 中文解释                               |
| -------- | ---------------------------- | ------------ | -------------------------------------- |
| EC       | Evolutionary Computation     | 进化计算       | 基于生物进化原理的优化和搜索方法。     |
| GA       | Genetic Algorithm            | 遗传算法       | 模拟自然选择和遗传机制的优化算法。     |
| Individual | Individual                   | 个体           | 问题的一个潜在解决方案。               |
| Population | Population                 | 种群           | 一组个体的集合。                     |
| Fitness Function | Fitness Function         | 适应度函数     | 评估个体解决问题能力的函数。         |
| Selection | Selection                    | 选择           | 根据适应度选择用于繁殖的个体。         |
| Crossover | Crossover                    | 交叉/杂交      | 结合两个或多个个体的基因产生新个体。   |
| Mutation | Mutation                     | 变异           | 随机改变个体基因的操作。               |
| Generation | Generation                 | 世代           | 种群的一次进化迭代。                 |
| Chromosome | Chromosome                   | 染色体         | 个体基因的编码表示。                 |
| Gene     | Gene                         | 基因           | 染色体上的一个特征或参数。           |
| Allele   | Allele                       | 等位基因       | 基因的不同可能取值。                 |
| Encoding | Encoding                     | 编码           | 将问题的解表示为染色体的过程。       |
| Decoding | Decoding                     | 解码           | 将染色体转换回问题解的过程。         |
| Termination Condition | Termination Condition    | 终止条件       | 算法停止运行的准则。                 |
| Evolutionary Strategy | Evolutionary Strategy (ES) | 进化策略       | 一种侧重于变异的进化算法。           |
| Genetic Programming | Genetic Programming (GP) | 遗传编程       | 进化计算机程序的进化算法。           |
| Particle Swarm Optimization | Particle Swarm Optimization (PSO) | 粒子群优化     | 模拟鸟群觅食行为的优化算法。         |
| Ant Colony Optimization | Ant Colony Optimization (ACO) | 蚁群优化       | 模拟蚂蚁觅食行为的优化算法。         |
| Biologically Inspired Computation | Biologically Inspired Computation | 生物启发计算     | 借鉴生物学原理的计算方法。           |


### 生物启发计算 (Biologically Inspired Computation)

| 英文简称 | 英文全称                     | 中文含义         | 中文解释                               |
| -------- | ---------------------------- | ------------ | -------------------------------------- |
| BIC      | Biologically Inspired Computation | 生物启发计算     | 借鉴生物学原理的计算方法。           |
| Evolutionary Computation | Evolutionary Computation (EC) | 进化计算       | 基于生物进化原理的优化和搜索方法。     |
| Genetic Algorithm | Genetic Algorithm (GA)   | 遗传算法       | 模拟自然选择和遗传机制的优化算法。     |
| Genetic Programming | Genetic Programming (GP) | 遗传编程       | 进化计算机程序的进化算法。           |
| Evolutionary Strategy | Evolutionary Strategy (ES) | 进化策略       | 一种侧重于变异的进化算法。           |
| Particle Swarm Optimization | Particle Swarm Optimization (PSO) | 粒子群优化     | 模拟鸟群或鱼群行为的优化算法。       |
| Ant Colony Optimization | Ant Colony Optimization (ACO) | 蚁群优化       | 模拟蚂蚁觅食行为的优化算法。         |
| Artificial Bee Colony | Artificial Bee Colony (ABC) | 人工蜂群算法 | 模拟蜜蜂采蜜行为的优化算法。         |
| Neural Networks | Neural Networks (NN)     | 神经网络       | 模拟生物神经元连接的网络模型。       |
| Deep Learning | Deep Learning (DL)       | 深度学习       | 使用多层神经网络进行复杂模式学习。       |
| Spiking Neural Networks | Spiking Neural Networks (SNN) | 脉冲神经网络   | 更接近生物神经元工作方式的神经网络。 |
| Cellular Automata | Cellular Automata (CA)   | 元胞自动机     | 基于局部规则的并行计算模型，模拟复杂系统。 |
| Artificial Immune Systems | Artificial Immune Systems (AIS) | 人工免疫系统   | 借鉴生物免疫系统原理的计算方法。     |
| DNA Computing | DNA Computing              | DNA计算       | 使用DNA分子进行计算。               |
| Membrane Computing | Membrane Computing       | 膜计算         | 基于细胞膜结构的计算模型。           |
| Swarm Robotics | Swarm Robotics           | 群体机器人     | 大量简单机器人协作完成任务。         |
| Morphogenetic Robotics | Morphogenetic Robotics   | 形态发生机器人 | 机器人能够自主改变自身形态。         |
| Natural Language Processing | Natural Language Processing (NLP) | 自然语言处理   | 部分方法借鉴人类语言处理机制。       |
| Computer Vision | Computer Vision (CV)     | 计算机视觉   | 部分模型借鉴生物视觉系统。           |
| Reinforcement Learning | Reinforcement Learning (RL) | 强化学习       | 借鉴动物通过奖励和惩罚学习行为的方式。 |
| Cognitive Science | Cognitive Science        | 认知科学       | 研究人类思维和智能，为AI提供灵感。   |


### 数据挖掘与知识发现 (Data Mining and Knowledge Discovery)

| 英文简称 | 英文全称                     | 中文含义         | 中文解释                               |
| -------- | ---------------------------- | ------------ | -------------------------------------- |
| DM       | Data Mining                  | 数据挖掘       | 从大量数据中发现有用模式和知识的过程。 |
| KDD      | Knowledge Discovery in Databases | 数据库知识发现 | 从数据库中发现知识的完整过程。         |
| Pattern  | Pattern                      | 模式           | 数据中存在的有意义的规律。             |
| Feature  | Feature                      | 特征           | 描述数据对象的属性。                 |
| Attribute | Attribute                    | 属性           | 与特征同义。                        |
| Instance | Instance                     | 实例           | 数据集中的一个数据对象。               |
| Data Preprocessing | Data Preprocessing       | 数据预处理     | 在挖掘前清洗和转换数据的步骤。       |
| Data Cleaning | Data Cleaning            | 数据清洗       | 处理数据中的缺失值、噪声和不一致性。 |
| Feature Selection | Feature Selection        | 特征选择       | 选择最相关的特征子集。               |
| Dimensionality Reduction | Dimensionality Reduction | 降维           | 减少数据的特征数量。                 |
| Clustering | Clustering                 | 聚类           | 将数据对象分组到相似的簇中。         |
| Classification | Classification           | 分类           | 将数据对象分配到预定义的类别。       |
| Regression | Regression                 | 回归           | 预测连续数值型变量。                 |
| Association Rule Mining | Association Rule Mining  | 关联规则挖掘 | 发现数据项之间的关联关系。           |
| Sequence Mining | Sequence Mining          | 序列模式挖掘   | 发现数据中按时间发生的模式。         |
| Outlier Detection | Outlier Detection        | 异常检测       | 识别与数据集中大多数对象不同的数据点。 |
| Data Visualization | Data Visualization       | 数据可视化     | 用图形方式展示数据和挖掘结果。       |
| Model Evaluation | Model Evaluation         | 模型评估       | 评估挖掘模型的性能。                 |
| Accuracy | Accuracy                     | 准确率       | 分类模型预测正确的比例。             |
| Precision | Precision                    | 精确率       | 在预测为正的样本中真正为正的比例。   |
| Recall  | Recall                       | 召回率       | 在所有正样本中被正确预测为正的比例。 |


### 信息检索 (Information Retrieval)

| 英文简称 | 英文全称                     | 中文含义     | 中文解释                               |
| -------- | ---------------------------- | -------- | -------------------------------------- |
| IR       | Information Retrieval        | 信息检索   | 从文档集合中找到相关信息的过程。       |
| Query    | Query                        | 查询       | 用户提出的信息需求。                   |
| Document | Document                     | 文档       | 信息检索系统处理的基本单元。           |
| Collection | Collection                   | 文档集合     | 信息检索系统处理的所有文档。           |
| Relevance | Relevance                    | 相关性       | 文档与用户查询的匹配程度。             |
| Ranking  | Ranking                      | 排序       | 根据相关性对检索到的文档进行排序。     |
| Indexing | Indexing                     | 索引       | 为文档集合创建快速查找结构的过程。     |
| Inverted Index | Inverted Index           | 倒排索引   | 一种常用的文档索引结构。             |
| Term Frequency | Term Frequency (TF)      | 词频       | 词语在文档中出现的次数。             |
| Inverse Document Frequency | Inverse Document Frequency (IDF) | 逆文档频率 | 衡量词语在整个文档集合中的重要性。   |
| TF-IDF   | Term Frequency-Inverse Document Frequency | 词频-逆文档频率 | 一种常用的词语权重计算方法。         |
| Precision | Precision                    | 精确率       | 返回的文档中相关的比例。               |
| Recall  | Recall                       | 召回率       | 所有相关文档中被返回的比例。         |
| F-measure | F-measure                    | F度量       | 精确率和召回率的调和平均值。         |
| MAP      | Mean Average Precision       | 平均精度均值 | 衡量排序结果质量的指标。             |
| NDCG     | Normalized Discounted Cumulative Gain | 归一化折损累积增益 | 考虑排序位置的评估指标。           |
| Query Expansion | Query Expansion          | 查询扩展     | 通过添加相关词语改进查询。           |
| Relevance Feedback | Relevance Feedback       | 相关性反馈   | 用户反馈用于改进检索结果。           |
| Web Search | Web Search                   | 网络搜索     | 在互联网上查找信息。                 |
| Search Engine | Search Engine              | 搜索引擎   | 实现网络搜索的系统。                 |


### 人机交互 (Human-Computer Interaction)

| 英文简称 | 英文全称                     | 中文含义         | 中文解释                               |
| -------- | ---------------------------- | ------------ | -------------------------------------- |
| HCI      | Human-Computer Interaction   | 人机交互       | 研究人与计算机系统之间交互的学科。     |
| UI       | User Interface               | 用户界面       | 用户与计算机系统交互的界面。           |
| UX       | User Experience              | 用户体验       | 用户在使用产品或服务时的整体感受。     |
| Usability | Usability                    | 易用性         | 用户使用系统的方便程度和效率。         |
| Interaction Design | Interaction Design     | 交互设计       | 设计用户与系统交互方式的学科。       |
| User-Centered Design | User-Centered Design   | 以用户为中心的设计 | 在设计过程中强调用户需求和反馈。       |
| Affordance | Affordance                   | 可供性         | 界面元素暗示其用途的特性。           |
| Feedback | Feedback                     | 反馈           | 系统对用户操作的响应。               |
| Mental Model | Mental Model               | 心理模型       | 用户对系统工作方式的理解。           |
| Heuristics | Heuristics                   | 启发式原则     | 设计易用界面的经验法则。             |
| Prototype | Prototype                    | 原型           | 系统的早期可交互版本。               |
| Evaluation | Evaluation                 | 评估           | 衡量系统易用性和用户体验的过程。       |
| User Testing | User Testing               | 用户测试       | 让真实用户使用系统并收集反馈。       |
| Heuristic Evaluation | Heuristic Evaluation     | 启发式评估     | 专家根据启发式原则评估界面。         |
| Cognitive Load | Cognitive Load             | 认知负荷       | 用户在操作时需要消耗的心理资源。     |
| Learnability | Learnability               | 易学性         | 用户学习使用系统的难易程度。         |
| Efficiency | Efficiency                 | 效率           | 用户完成任务所需的时间和资源。       |
| Memorability | Memorability               | 易记性         | 用户再次使用时记住操作的难易程度。   |
| Errors   | Errors                       | 错误           | 用户在操作过程中发生的失误。         |
| Satisfaction | Satisfaction               | 满意度         | 用户对系统的整体感觉。               |


### 机器翻译 (Machine Translation)

| 英文简称 | 英文全称                     | 中文含义       | 中文解释                               |
| -------- | ---------------------------- | ---------- | -------------------------------------- |
| MT       | Machine Translation          | 机器翻译     | 使用计算机程序自动将文本从一种语言翻译成另一种语言。 |
| SMT      | Statistical Machine Translation | 统计机器翻译 | 基于统计模型和大量双语语料库的机器翻译方法。 |
| NMT      | Neural Machine Translation   | 神经机器翻译 | 基于神经网络（尤其是深度学习模型）的机器翻译方法。 |
| RBMT     | Rule-Based Machine Translation | 基于规则的机器翻译 | 基于预定义的语言学规则进行翻译的方法。 |
| EBMT     | Example-Based Machine Translation | 基于实例的机器翻译 | 基于已有的翻译实例进行类比翻译的方法。 |
| Word Alignment | Word Alignment           | 词对齐       | 在双语语料库中标记源语言和目标语言词语之间的对应关系。 |
| Bilingual Corpus | Bilingual Corpus         | 双语语料库     | 包含源语言文本及其对应目标语言翻译的文本集合。 |
| Parallel Corpus | Parallel Corpus          | 平行语料库     | 与双语语料库同义。                     |
| Translation Memory | Translation Memory       | 翻译记忆     | 存储已翻译过的文本片段及其译文，用于加速后续翻译。 |
| BLEU     | Bilingual Evaluation Understudy | 双语评估替补 | 一种常用的机器翻译质量自动评估指标。 |
| METEOR   | Metric for Evaluation of Translation with Explicit Ordering | METEOR       | 另一种机器翻译质量自动评估指标，考虑了词形和同义词。 |
| TER      | Translation Edit Rate        | 翻译编辑率   | 衡量将机器翻译结果修改为人工翻译所需的编辑操作数量。 |
| Source Language | Source Language          | 源语言       | 需要被翻译的语言。                   |
| Target Language | Target Language          | 目标语言       | 翻译后的语言。                       |
| Decoding | Decoding                     | 解码         | 在神经机器翻译中，将编码后的语义表示生成目标语言文本的过程。 |
| Attention Mechanism | Attention Mechanism      | 注意力机制     | 神经机器翻译中的一种技术，使模型能够关注输入序列中最相关的部分。 |
| Encoder-Decoder Architecture | Encoder-Decoder Architecture | 编码器-解码器架构 | 神经机器翻译中常用的模型结构。         |
| Transfer Learning in MT | Transfer Learning in MT | 机器翻译中的迁移学习 | 将在一个语言对上训练的模型应用于另一个语言对。 |
| Low-Resource MT | Low-Resource MT          | 低资源机器翻译 | 针对缺乏充足双语语料库的语言对进行机器翻译。 |
| Post-Editing | Post-Editing               | 译后编辑     | 人工修改机器翻译结果以提高质量。       |


### 因果推断 (Causal Inference)

| 英文简称 | 英文全称                     | 中文含义     | 中文解释                               |
| -------- | ---------------------------- | -------- | -------------------------------------- |
| CI       | Causal Inference           | 因果推断     | 识别变量之间因果关系的过程。           |
| Causation | Causation                    | 因果关系     | 一个事件（原因）导致另一个事件（结果）。 |
| Correlation | Correlation                | 相关性       | 两个或多个变量之间存在统计关联。       |
| Intervention | Intervention               | 干预         | 对某个变量进行人为改变。             |
| Treatment | Treatment                    | 处理         | 被干预的变量或条件。                 |
| Outcome  | Outcome                      | 结果         | 我们希望观察其是否受处理影响的变量。   |
| Confounding Variable | Confounding Variable   | 混淆变量     | 同时影响处理和结果的外部变量。       |
| Randomization | Randomization              | 随机化       | 随机分配处理以消除混淆变量的影响。     |
| Control Group | Control Group              | 对照组       | 未接受处理的组，用于比较。           |
| Treatment Group | Treatment Group          | 处理组       | 接受处理的组。                       |
| Potential Outcome | Potential Outcome        | 潜在结果     | 在不同处理条件下可能发生的结果。     |
| Average Treatment Effect | Average Treatment Effect (ATE) | 平均处理效应 | 处理对总体平均结果的影响。           |
| Conditional Average Treatment Effect | Conditional Average Treatment Effect (CATE) | 条件平均处理效应 | 处理对特定人群平均结果的影响。       |
| Instrumental Variable | Instrumental Variable (IV) | 工具变量     | 与处理相关但不直接影响结果的变量，用于识别因果效应。 |
| Regression Discontinuity Design | Regression Discontinuity Design (RDD) | 回归不连续性设计 | 利用人为设定的阈值进行因果推断。     |
| Difference-in-Differences | Difference-in-Differences (DID) | 倍差法       | 比较处理组和对照组在处理前后结果的差异。 |
| Propensity Score Matching | Propensity Score Matching (PSM) | 倾向得分匹配 | 匹配具有相似倾向得分的处理组和对照组。 |
| Causal Graph | Causal Graph               | 因果图       | 用图结构表示变量及其因果关系。       |
| Directed Acyclic Graph | Directed Acyclic Graph (DAG) | 有向无环图   | 一种表示因果关系的图模型。         |
| Counterfactual | Counterfactual             | 反事实       | 与实际发生情况相反的假设情景。       |


### AI音乐 (AI Music)

| 英文简称 | 英文全称                     | 中文含义       | 中文解释                               |
| -------- | ---------------------------- | ---------- | -------------------------------------- |
| AI Music | AI Music                   | 人工智能音乐 | 由人工智能算法生成的音乐作品。       |
| Music Generation | Music Generation         | 音乐生成     | 使用AI模型创作音乐的过程。           |
| Algorithmic Composition | Algorithmic Composition | 算法作曲     | 使用算法和规则自动创作音乐。         |
| RNN      | Recurrent Neural Network     | 循环神经网络 | 常用于生成序列数据的音乐。           |
| LSTM     | Long Short-Term Memory       | 长短期记忆网络 | 擅长处理音乐中的长期依赖关系。       |
| Transformer | Transformer                | Transformer  | 基于自注意力机制的音乐生成模型。       |
| GAN for Music | GAN for Music              | GAN用于音乐  | 使用生成对抗网络生成音乐。           |
| Style Transfer in Music | Style Transfer in Music | 音乐风格迁移 | 将一种音乐的风格应用于另一种音乐。   |
| Music Information Retrieval | Music Information Retrieval (MIR) | 音乐信息检索 | 从音乐数据中提取和分析信息。       |
| Human-AI Collaboration in Music | Human-AI Collaboration in Music | 人机协作音乐 | 音乐家与AI共同创作音乐。           |
| Interactive Music | Interactive Music        | 互动音乐     | 音乐可以根据听众或环境变化。         |


### 自动推理 (Automated Reasoning)

| 英文简称 | 英文全称                     | 中文含义     | 中文解释                               |
| -------- | ---------------------------- | -------- | -------------------------------------- |
| AR       | Automated Reasoning          | 自动推理   | 使用计算机程序进行逻辑推理。           |
| Logic    | Logic                        | 逻辑       | 研究有效推理的原理和规则。             |
| Inference Rule | Inference Rule           | 推理规则   | 从已知事实推导新事实的规则。           |
| Theorem Proving | Theorem Proving          | 定理证明   | 使用形式逻辑证明数学定理。           |
| Resolution | Resolution                   | 消解       | 一种用于定理证明的推理规则。           |
| Unification | Unification                | 合一       | 寻找使两个表达式相等的最一般代换。     |
| First-Order Logic | First-Order Logic (FOL)  | 一阶逻辑   | 一种强大的形式逻辑系统。             |
| Propositional Logic | Propositional Logic      | 命题逻辑   | 一种基本的形式逻辑系统。             |
| Knowledge Representation | Knowledge Representation | 知识表示   | 如何在计算机中表达和存储知识。         |
| Ontology | Ontology                     | 知识本体   | 对特定领域概念及其关系的明确规范。     |
| Semantic Web | Semantic Web               | 语义网     | 使网络数据更具机器可理解性的技术集合。 |
| Knowledge Graph  | Knowledge Graph          | 知识图谱   | 由实体和关系组成的图状知识库。       |
| Rule-Based System | Rule-Based System        | 基于规则系统 | 使用规则进行推理的系统。               |
| Expert System | Expert System            | 专家系统   | 模拟人类专家解决问题的系统。         |
| Forward Chaining | Forward Chaining         | 前向推理   | 从已知事实推导结论。                 |
| Backward Chaining | Backward Chaining        | 反向推理   | 从目标反向寻找支持证据。             |
| Abductive Reasoning | Abductive Reasoning      | 溯因推理   | 从观察到的结果推断最可能的解释。     |
| Inductive Reasoning | Inductive Reasoning      | 归纳推理   | 从个别案例推广到一般规律。           |
| Deductive Reasoning | Deductive Reasoning      | 演绎推理   | 从一般规律推导出个别结论。           |
| Constraint Satisfaction | Constraint Satisfaction (CSP) | 约束满足     | 寻找满足约束条件的解。               |


### 知识图谱 (Knowledge Graphs)

| 英文简称 | 英文全称                     | 中文含义       | 中文解释                               |
| -------- | ---------------------------- | ---------- | -------------------------------------- |
| KG       | Knowledge Graph            | 知识图谱       | 由实体和关系组成的图状知识库。       |
| Node     | Node                         | 节点         | 图谱中的实体或概念。                   |
| Edge     | Edge                         | 边           | 节点之间的关系。                     |
| Entity   | Entity                       | 实体         | 图谱中代表现实世界对象的节点。         |
| Relation | Relation                     | 关系         | 连接两个实体的边，描述它们之间的联系。 |
| Attribute | Attribute                    | 属性         | 实体的特征或属性值。                 |
| Ontology | Ontology                     | 知识本体       | 对领域概念和关系的明确规范。         |
| Schema   | Schema                       | 图谱模式     | 知识图谱的结构和组织框架。           |
| Triple   | Triple                       | 三元组       | 实体-关系-实体（或实体-属性-值）的基本单元。 |
| RDF      | Resource Description Framework | 资源描述框架   | 常用于表示知识图谱的标准。           |
| SPARQL   | SPARQL Protocol and RDF Query Language | SPARQL       | 查询知识图谱的语言。                  |
| Graph Database | Graph Database           | 图数据库     | 专门用于存储和查询图结构数据的数据库。 |
| Semantic Search | Semantic Search          | 语义搜索     | 基于知识图谱理解查询意图的搜索。     |
| Entity Linking | Entity Linking           | 实体链接     | 将文本中的提及项链接到知识图谱中的实体。 |
| Relation Extraction | Relation Extraction      | 关系抽取     | 从文本中识别实体之间的关系。         |
| Knowledge Graph Embedding | Knowledge Graph Embedding | 知识图谱嵌入 | 将实体和关系表示为低维向量。         |
| Knowledge Graph Completion | Knowledge Graph Completion | 知识图谱补全 | 推断知识图谱中缺失的实体或关系。     |
| Reasoning  | Reasoning                    | 推理         | 基于知识图谱中的信息推导新知识。       |
| Knowledge Acquisition | Knowledge Acquisition    | 知识获取     | 从不同来源构建知识图谱的过程。       |
| Data Integration | Data Integration         | 数据集成     | 合并来自不同来源的数据到知识图谱。   |
| Named Entity Recognition | Named Entity Recognition (NER) | 命名实体识别 | 识别文本中具有特定意义的实体。         |


### 评估指标（Evaluation metrics）

| 英文简称 | 英文全称                          | 中文翻译         | 解释说明                     |
|----------|-----------------------------------|----------------|----------------------------|
| ROC      | Receiver Operating Characteristic | 受试者工作特征曲线 | 展示分类模型性能的图形     |
| F1       | F1 Score                          | F1分数         | 精确率与召回率的调和平均     |
| Precision | Precision                       | 精确率         | 预测为正样本中的真实正样本比 |
| Recall   | Recall                           | 召回率         | 真实正样本中被正确预测的比   |
| MSE      | Mean Squared Error                | 均方误差       | 回归任务的平均预测误差平方   |
| MAE      | Mean Absolute Error               | 平均绝对误差   | 回归任务的平均绝对误差       |
| BLEU     | Bilingual Evaluation Understudy   | 双语评估研究   | 机器翻译质量评估指标         |
| Perplexity | Perplexity                     | 困惑度         | 语言模型预测能力的衡量指标   |
| Rouge    | Recall-Oriented Understudy for Gisting Evaluation | ROUGE评分     | 文本生成任务的质量评估指标   |
| IoU      | Intersection over Union           | 交并比         | 目标检测中预测框与真实框重叠度 |
| PSNR     | Peak Signal-to-Noise Ratio        | 峰值信噪比     | 图像重建质量的客观评价指标   |
