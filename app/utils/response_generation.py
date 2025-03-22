"""
Response generation utilities for the NGSS Curriculum Coach

This module provides functions for generating coach-like responses
based on retrieved curriculum resources.
"""

import os
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOpenAI


class ResponseGenerator:
    """Class for generating coach-like responses"""
    
    def __init__(self, prompts_dir="prompts", temperature=0.7):
        """Initialize the response generator"""
        self.prompts_dir = prompts_dir
        self.llm = ChatOpenAI(temperature=temperature)
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
            "lesson_finder": """SPECIFIC INSTRUCTIONS FOR FINDING LESSONS BY DCI OR PHENOMENON...""",
            "sep_identifier": """SPECIFIC INSTRUCTIONS FOR IDENTIFYING SCIENCE AND ENGINEERING PRACTICES...""",
            "model_analyzer": """SPECIFIC INSTRUCTIONS FOR ANALYZING MODEL DEVELOPMENT...""",
            "assessment_mapper": """SPECIFIC INSTRUCTIONS FOR IDENTIFYING ASSESSMENTS...""",
            "general_guidance": """GENERAL CURRICULUM GUIDANCE INSTRUCTIONS..."""
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
        start_idx = max(0, len(chat_history) - (max_turns * 2))
        for i in range(start_idx, len(chat_history), 2):
            if i < len(chat_history):
                history_str += f"Teacher: {chat_history[i]['content']}\n"
            if i+1 < len(chat_history):
                history_str += f"Coach: {chat_history[i+1]['content']}\n"
        return history_str
    
    def generate_response(self, query, intent, retrieved_docs, user_context=None, chat_history=None):
        """Generate a coach-like response based on intent and retrieved documents"""
        user_context = user_context or {}
        chat_history = chat_history or []
        curriculum_context = self.format_retrieved_context(retrieved_docs)
        teacher_context = self.format_user_context(user_context)
        history_context = self.format_chat_history(chat_history)
        intent_template = self.templates.get(intent, self.templates["general_guidance"])
        system_template = f"{self.templates['system']}\n\n{intent_template}"
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = """{teacher_context}\n\n{history_context}\n\nTEACHER QUERY: {query}\n\n{curriculum_context}\n\nBased on the curriculum resources and the teacher's query, provide a helpful, supportive response that offers specific guidance and recommendations. Maintain a coach-like, encouraging tone throughout."""
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        response = self.llm(chat_prompt.format_messages(
            teacher_context=teacher_context,
            history_context=history_context,
            query=query,
            curriculum_context=curriculum_context
        ))
        return response.content
