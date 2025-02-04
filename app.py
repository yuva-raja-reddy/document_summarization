import gradio as gr
import os
import random
import nltk
import networkx as nx
import numpy as np
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize

# Download NLTK dependencies
nltk.download('punkt_tab')

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def read_article(text):        
    sentences = sent_tokenize(text)    
    return sentences

def sentence_similarity(sent1, sent2, embed):  
    A = embed([sent1])[0]
    B = embed([sent2])[0]
    return 1 - (np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)))

def build_similarity_matrix(sentences, embeds):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], embeds)
    return similarity_matrix

def generate_summary(text, top_n):
    summarize_text = []  
    sentences = read_article(text)           
    sentence_similarity_matrix = build_similarity_matrix(sentences, embed)  
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph) 
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    for i in range(min(top_n, len(ranked_sentences))):
        summarize_text.append(ranked_sentences[i][1]) 
    return " ".join(summarize_text)

def summarize(text, num_sentences):
    return generate_summary(text, num_sentences)

random_texts = [
    """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals and humans. AI research explores problem-solving techniques such as reasoning, perception, and knowledge representation. Over the years, advancements in AI have led to significant progress in areas like speech recognition, computer vision, and robotics. Machine learning, a subset of AI, enables systems to learn from data and make predictions. Deep learning, a more advanced technique, uses neural networks to detect complex patterns. AI applications range from self-driving cars to automated medical diagnostics. However, AI also raises ethical concerns regarding bias, privacy, and automation's impact on employment. Policymakers are working on frameworks to regulate AI while maximizing its benefits. The future of AI includes advancements in artificial general intelligence (AGI), which aims to mimic human intelligence across tasks. Companies and researchers are working towards more responsible AI development. AI-driven automation is transforming workplaces, introducing both opportunities and challenges. AI chatbots and virtual assistants are becoming increasingly common in customer service. Understanding AI’s limitations is crucial to prevent misinformation and misuse. AI ethics focuses on ensuring that models make fair and unbiased decisions. AI is also being integrated into creative fields, generating music, art, and writing. Future AI systems could be capable of self-learning without human intervention. The debate on AI regulation continues as technology evolves. AI in healthcare is helping doctors diagnose diseases more accurately. Companies are investing billions in AI research to stay competitive. The rapid development of AI requires continuous monitoring and evaluation to align with human values.""",
    
    """Climate change is a long-term shift in weather patterns, primarily driven by human activities such as burning fossil fuels. The increase in greenhouse gases traps heat, leading to rising global temperatures. Scientists have observed a drastic increase in extreme weather events, including hurricanes, droughts, and wildfires. Melting polar ice caps are causing sea levels to rise, threatening coastal communities. Climate change also affects biodiversity, leading to habitat loss for many species. Agriculture is heavily impacted, with unpredictable weather patterns affecting crop yields. Governments worldwide are implementing policies to reduce carbon emissions and promote renewable energy. The Paris Agreement is a global effort to limit temperature rise and mitigate climate effects. Public awareness and individual actions, like reducing energy consumption, are crucial in combating climate change. Scientists are developing carbon capture technology to remove CO2 from the atmosphere. The shift to electric vehicles is reducing dependency on fossil fuels. Deforestation contributes to climate change by reducing carbon absorption from trees. Many companies are adopting sustainable practices to minimize their carbon footprint. Renewable energy sources like solar and wind are becoming more accessible and affordable. Climate activists advocate for stronger regulations and accountability from large industries. The global economy is adapting to more sustainable production methods. Education plays a key role in spreading awareness about climate change. Governments are investing in climate resilience infrastructure to protect vulnerable areas. The future of climate action depends on technological innovation and international cooperation.""",
    
    """The history of space exploration began in the mid-20th century with the launch of Sputnik 1 by the Soviet Union. This marked the beginning of the space race between the USA and USSR. The Apollo program led to the first human landing on the Moon in 1969, a defining moment in history. Since then, advancements in space technology have enabled robotic missions to explore Mars, Jupiter, and beyond. The Hubble Space Telescope has provided breathtaking images of the universe, helping astronomers understand cosmic phenomena. The International Space Station (ISS) serves as a collaborative research hub for astronauts from multiple countries. Private companies like SpaceX and Blue Origin are revolutionizing space travel with reusable rocket technology. Plans for manned missions to Mars are actively being developed, aiming for the 2030s. Space tourism is becoming a reality, with civilians now able to experience space travel. The search for extraterrestrial life continues, with rovers analyzing Martian soil for signs of microbial life. Scientists study exoplanets to determine their potential habitability. Deep-space exploration is unlocking secrets of black holes, dark matter, and the origins of the universe. The future of space travel includes plans for lunar bases and asteroid mining. Advancements in satellite technology are improving global communications and Earth monitoring. NASA and other space agencies are collaborating on ambitious interstellar projects. Space junk poses a growing problem, requiring better debris management strategies. The potential for space colonization is a topic of scientific debate and ethical consideration. Cutting-edge propulsion systems could one day allow interstellar travel. Artificial intelligence is being integrated into space missions to enhance efficiency. The discovery of water on Mars has renewed interest in the planet's potential for sustaining life.""",
    
    """The human brain is one of the most complex structures known to science. It contains billions of neurons that communicate through electrical and chemical signals. Brain function is responsible for cognition, emotions, and motor control. Neuroscientists study the brain to understand consciousness and memory formation. Brain plasticity allows neurons to rewire and adapt to new experiences. Damage to certain brain areas can lead to cognitive disorders like Alzheimer's and Parkinson's. The field of neurotechnology is developing brain-machine interfaces that enable communication through thought. Brain scans like MRI and fMRI help researchers study brain activity in real-time. Sleep is crucial for brain function, allowing memory consolidation and cognitive restoration. Certain neurotransmitters like dopamine and serotonin regulate mood and emotions. The blood-brain barrier protects the brain from harmful substances but also limits drug delivery. Meditation and mindfulness practices have been shown to positively affect brain structure. Neurological research aims to develop treatments for conditions like depression and epilepsy. Understanding brain function could one day lead to artificial intelligence mimicking human thought processes. The debate on free will and consciousness continues in philosophical and scientific discussions. The study of dreams provides insight into subconscious thought processing. Brain injuries can have long-lasting effects on personality and cognition. Cognitive enhancement through brain stimulation is an emerging field of research. New discoveries in neuroscience continue to shape our understanding of human intelligence and behavior.""",
    
    """Blockchain technology has revolutionized digital transactions by providing secure, decentralized record-keeping. Initially developed for Bitcoin, blockchain is now used in finance, healthcare, and logistics. The core principle of blockchain is a distributed ledger system that ensures transparency and immutability. Smart contracts automate transactions without the need for intermediaries. Cryptocurrencies like Ethereum and Bitcoin operate on blockchain networks. Decentralized finance (DeFi) is expanding financial opportunities without traditional banks. Non-fungible tokens (NFTs) use blockchain to verify ownership of digital assets. Governments are exploring the use of blockchain for secure voting systems. The security of blockchain relies on cryptographic hashing and consensus mechanisms. While highly secure, blockchain faces challenges like scalability and energy consumption. Environmental concerns have led to the development of eco-friendly consensus models. The technology is also being explored for supply chain transparency and fraud prevention. Large corporations are investing in blockchain to streamline business operations. Regulators are working to establish laws governing cryptocurrency and blockchain applications. Cross-border payments using blockchain reduce transaction fees and increase speed. Blockchain-based identity management could help eliminate identity theft. The concept of Web3 envisions a fully decentralized internet powered by blockchain. Quantum computing poses a potential threat to blockchain encryption in the future. Innovations in layer 2 scaling solutions aim to enhance blockchain efficiency. The continued evolution of blockchain is expected to transform multiple industries worldwide."""
]

def get_random_text():
    return random.choice(random_texts)

def process_input(user_input, num_sentences):
    return summarize(user_input, num_sentences)

def set_random_text():
    return get_random_text()

def clear_fields():
    return "", ""

demo = gr.Blocks()
with demo:
    with gr.Row():
        user_input = gr.Textbox(lines=10, placeholder="Enter text to summarize or click Generate Random Text")
    
    num_sentences = gr.Slider(1, 10, step=1, label="Number of sentences")
    output = gr.Textbox(label="Output")
    
    with gr.Row():
        generate_btn = gr.Button("Generate Random Text")
        submit_btn = gr.Button("Summarize")
        clear_btn = gr.Button("Clear")
    
    generate_btn.click(set_random_text, inputs=[], outputs=[user_input])
    submit_btn.click(process_input, inputs=[user_input, num_sentences], outputs=[output])
    clear_btn.click(clear_fields, inputs=[], outputs=[user_input, output])

demo.launch()
