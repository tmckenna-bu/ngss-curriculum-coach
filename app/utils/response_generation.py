"""
Response generation utilities for the NGSS Curriculum Coach

This module provides functions for generating coach-like responses
based on retrieved curriculum resources.
"""

import os
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOpenAI
import streamlit as st

class ResponseGenerator:
    """Class for generating coach-like responses"""
    
    def __init__(self, prompts_dir="prompts", temperature=0.7):
        """Initialize the response generator"""
        self.prompts_dir = prompts_dir
        
        # Try to initialize OpenAI, but gracefully handle errors
        try:
            self.llm = ChatOpenAI(temperature=temperature)
            self.openai_available = True
        except Exception as e:
            st.warning("OpenAI API not available. Running in demo mode with pre-written responses.")
            self.openai_available = False
            
        self.templates = self._load_templates()
    
    def _load_templates(self):
        """Load prompt templates from files"""
        templates = {}
        
        # Ensure prompts directory exists
        if not os.path.exists(self.prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        # System prompt (required)
        system_prompt_path = os.path.join(self.prompts_dir, "system_prompt.txt")
        if not os.path.exists(system_prompt_path):
            raise FileNotFoundError(f"System prompt not found: {system_prompt_path}")
        
        with open(system_prompt_path, "r") as f:
            templates["system"] = f.read()
        
        # Load other templates if available
        template_files = {
            "lesson_finder": "lesson_finder.txt",
            "sep_identifier": "sep_identifier.txt",
            "model_analyzer": "model_analyzer.txt",
            "assessment_mapper": "assessment_mapper.txt",
            "general_guidance": "general_guidance.txt"
        }
        
        for key, filename in template_files.items():
            file_path = os.path.join(self.prompts_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    templates[key] = f.read()
            else:
                # Use default templates as fallback
                templates[key] = self._get_default_template(key)
        
        return templates
    
    def _get_default_template(self, template_key):
        """Get default templates as fallback"""
        defaults = {
            "lesson_finder": "SPECIFIC INSTRUCTIONS FOR FINDING LESSONS BY DCI OR PHENOMENON...",
            "sep_identifier": "SPECIFIC INSTRUCTIONS FOR IDENTIFYING SCIENCE AND ENGINEERING PRACTICES...",
            "model_analyzer": "SPECIFIC INSTRUCTIONS FOR ANALYZING MODEL DEVELOPMENT...",
            "assessment_mapper": "SPECIFIC INSTRUCTIONS FOR IDENTIFYING ASSESSMENTS...",
            "general_guidance": "GENERAL CURRICULUM GUIDANCE INSTRUCTIONS..."
        }
        
        return defaults.get(template_key, "")
    
    def format_retrieved_context(self, docs):
        """Format retrieved documents as context for the LLM"""
        if not docs:
            return "No curriculum resources found."
            
        formatted_context = "RETRIEVED CURRICULUM RESOURCES:\n\n"
        
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            formatted_context += f"--- DOCUMENT {i+1} ---\n"
            formatted_context += f"Title: {metadata.get('title', 'Untitled')}\n"
            
            if 'grade_level' in metadata and metadata['grade_level'] is not None:
                formatted_context += f"Grade: {metadata['grade_level']}\n"
                
            if 'unit' in metadata and metadata['unit']:
                formatted_context += f"Unit: {metadata['unit']}\n"
                
            if 'dci' in metadata and metadata['dci']:
                formatted_context += f"DCIs: {', '.join(metadata['dci'])}\n"
                
            if 'sep' in metadata and metadata['sep']:
                formatted_context += f"SEPs: {', '.join(metadata['sep'])}\n"
                
            if 'ccc' in metadata and metadata['ccc']:
                formatted_context += f"CCCs: {', '.join(metadata['ccc'])}\n"
                
            if 'phenomena' in metadata and metadata['phenomena']:
                formatted_context += f"Phenomena: {', '.join(metadata['phenomena'])}\n"
                
            if 'assessment_types' in metadata and metadata['assessment_types']:
                formatted_context += f"Assessment Types: {', '.join(metadata['assessment_types'])}\n"
                
            formatted_context += f"\nContent: {doc.page_content}\n\n"
            
        return formatted_context
    
    def format_user_context(self, user_context):
        """Format user context for the LLM"""
        context_str = "TEACHER CONTEXT:\n"
        
        if user_context.get('grade_level'):
            context_str += f"- The teacher teaches grade {user_context['grade_level']}\n"
            
        if user_context.get('previous_topics'):
            context_str += f"- Previously discussed topics: {', '.join(user_context['previous_topics'])}\n"
            
        if user_context.get('mentioned_challenges'):
            context_str += f"- Mentioned challenges: {', '.join(user_context['mentioned_challenges'])}\n"
            
        return context_str
    
    def format_chat_history(self, chat_history, max_turns=3):
        """Format recent chat history"""
        if not chat_history:
            return ""
            
        history_str = "RECENT CONVERSATION:\n"
        
        # Get last few turns (each turn is a user message and assistant response)
        start_idx = max(0, len(chat_history) - (max_turns * 2))
        
        for i in range(start_idx, len(chat_history), 2):
            if i < len(chat_history):
                history_str += f"Teacher: {chat_history[i]['content']}\n"
            if i+1 < len(chat_history):
                history_str += f"Coach: {chat_history[i+1]['content']}\n"
        
        return history_str
    
    def generate_response(self, query, intent, retrieved_docs, user_context=None, chat_history=None):
        """Generate a coach-like response based on intent and retrieved documents"""
        # Set defaults
        if user_context is None:
            user_context = {}
        if chat_history is None:
            chat_history = []
        
        # Format context
        curriculum_context = self.format_retrieved_context(retrieved_docs)
        teacher_context = self.format_user_context(user_context)
        history_context = self.format_chat_history(chat_history)
        
        # If OpenAI is not available, return a demo response
        if not self.openai_available:
            return self._get_demo_response(intent, query)
        
        # Select template based on intent
        if intent in self.templates:
            intent_template = self.templates[intent]
        else:
            intent_template = self.templates["general_guidance"]
        
        # Create system message with base prompt and intent-specific instructions
        system_template = f"{self.templates['system']}\n\n{intent_template}"
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        # Create human message with context and query
        human_template = """
        {teacher_context}

        {history_context}

        TEACHER QUERY: {query}

        {curriculum_context}

        Based on the curriculum resources and the teacher's query, provide a helpful, supportive response that offers specific guidance and recommendations. Maintain a coach-like, encouraging tone throughout.
        """
        
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        # Create chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        try:
            # Generate response
            response = self.llm.invoke(chat_prompt.format_messages(
                teacher_context=teacher_context,
                history_context=history_context,
                query=query,
                curriculum_context=curriculum_context
            ))
            
            return response.content
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return self._get_demo_response(intent, query)
    
    def _get_demo_response(self, intent, query):
        """Provide a pre-written response when OpenAI is not available"""
        demo_responses = {
            "lesson_finder": f"""
            I'd be happy to help you find lessons related to "{query}"!
            
            Based on our curriculum resources, I recommend:
            
            **1. Energy Transfer in Ecosystems (Grade 6)**
            This unit explores how energy flows through ecosystems, with hands-on activities where students track energy transfer. It addresses PS3.D and LS2.B DCIs and includes both modeling and explanation activities.
            
            **2. Interactions Investigation (Grade 9)**
            This unit examines interactions between particles and energy, with engaging phenomena like static electricity demonstrations. Students develop models to explain electrostatic interactions.
            
            Would you like more details about either of these options?
            """,
            
            "sep_identifier": f"""
            Looking at the activity you mentioned in "{query}", I can identify several key Science and Engineering Practices:
            
            **Primary SEP: Developing and Using Models**
            Students create visual representations of energy flow and revise these models based on new evidence. This is demonstrated when they construct food web diagrams.
            
            **Secondary SEP: Constructing Explanations**
            Students use evidence from their observations to explain the relationships between organisms in the ecosystem.
            
            To enhance these practices, consider having students critique each other's models in small groups, which adds the "Engaging in Argument from Evidence" practice.
            """,
            
            "model_analyzer": f"""
            Regarding your question about modeling in "{query}", the unit supports model development through a carefully scaffolded progression:
            
            **Initial Models (Lesson 1):**
            Students construct initial models explaining their current understanding of the phenomenon.
            
            **Model Revision (Lessons 2-3):**
            As students gather evidence through investigations, they revise their models to incorporate new understandings.
            
            **Final Explanatory Models (Lesson 4):**
            Students develop comprehensive models that explain the targeted phenomenon.
            
            To support diverse learners, consider providing visual scaffolds and sentence stems for model annotations.
            """,
            
            "assessment_mapper": f"""
            The unit you're asking about includes several assessment opportunities:
            
            **Formative Assessments:**
            - Exit tickets in lessons 1 and 2 to check understanding of key concepts
            - Group discussion monitoring during modeling activities
            
            **Performance Assessments:**
            - Model development and revision throughout the unit
            - Final explanation writing task that addresses the performance expectation
            
            Each assessment is aligned to specific DCIs and SEPs, providing a comprehensive picture of student understanding.
            """,
            
            "general_guidance": f"""
            Thank you for your question about "{query}"! 
            
            While I don't have access to the OpenAI API right now to generate a personalized response, I can tell you that our NGSS-aligned curriculum is designed to integrate the three dimensions of science learning: Disciplinary Core Ideas, Science and Engineering Practices, and Crosscutting Concepts.
            
            The resources we have focus on phenomena-based instruction and provide multiple opportunities for students to engage in science practices while developing deep understanding of core ideas.
            
            Would you like to know more about specific units, practices, or assessment strategies?
            """
        }
        
        return demo_responses.get(intent, demo_responses["general_guidance"])
