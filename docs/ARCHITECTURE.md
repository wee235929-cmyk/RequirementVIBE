# ReqVibe Architecture

## Overview

This document describes the architectural organization of the ReqVibe project, following principles of **high cohesion** and **low coupling**.

## Architecture Layers

The project is organized into clear architectural layers:

```
RequirenebtVIBE/
├── core/                    # Core domain models and shared abstractions
│   └── models/              # Shared domain models (e.g., ShortTermMemory)
│
├── domain/                  # Domain-specific business logic
│   ├── requirements/        # Requirements extraction and management
│   ├── conversations/       # Conversation storage and management
│   ├── sessions/            # Session management
│   ├── prompts/             # Prompt generation and template rendering
│   │   └── templates/       # Jinja2 templates
│   └── documents/           # Document processing (SRS, Unstructured API)
│
├── application/             # Application services (orchestration layer)
│   └── auth/                # Authentication services
│
├── infrastructure/          # External integrations
│   ├── llm/                 # LLM client integrations
│   ├── graphrag/            # GraphRAG knowledge graph services
│   ├── tools/               # Tool integrations (Mermaid, Gherkin)
│   └── voice/               # Voice transcription (Whisper)
│
├── presentation/            # UI layer (Streamlit)
│   ├── components/          # Reusable UI components
│   └── styles.py            # CSS styling
│
├── config/                  # Configuration
│   ├── models.py            # Configuration models
│   └── roles/               # Role configurations (JSON)
│
└── utils/                   # Shared utilities
    ├── state_manager.py     # Session state management
    └── renderers/           # Rendering utilities (Mermaid)
```

## Design Principles

### 1. High Cohesion
- **Domain Services**: Related functionality grouped by domain (requirements, conversations, sessions, prompts, documents)
- **Infrastructure**: External integrations grouped by type (LLM, GraphRAG, tools)
- **Presentation**: UI components organized by purpose (components, pages, styles)

### 2. Low Coupling
- **Layer Independence**: Each layer depends only on layers below it
- **Domain Isolation**: Domain services don't depend on infrastructure or presentation
- **Interface-Based**: Services communicate through well-defined interfaces

### 3. Separation of Concerns
- **Core**: Shared models and abstractions
- **Domain**: Business logic and domain rules
- **Application**: Orchestration and use cases
- **Infrastructure**: External system integrations
- **Presentation**: User interface and interaction

## Import Guidelines

### Allowed Dependencies
- **Presentation** → Domain, Application, Infrastructure, Core, Utils
- **Application** → Domain, Infrastructure, Core, Utils
- **Domain** → Core, Utils (no infrastructure or presentation dependencies)
- **Infrastructure** → Core, Utils (no domain dependencies)
- **Core** → Utils only
- **Utils** → No dependencies on other layers

### Import Examples
```python
# ✅ Good: Presentation importing from Domain
from domain.sessions.service import create_new_session

# ✅ Good: Domain importing from Core
from core.models.memory import ShortTermMemory

# ❌ Bad: Domain importing from Infrastructure
from infrastructure.llm.client import get_deepseek_client  # Use dependency injection instead

# ❌ Bad: Domain importing from Presentation
import streamlit as st  # Use dependency injection or pass as parameter
```

## Module Organization

### Domain Services
Each domain has its own directory with:
- `service.py`: Main service implementation
- `__init__.py`: Public API exports

### Infrastructure Services
External integrations are isolated in the infrastructure layer:
- **LLM**: Centralized API client wrapper
- **GraphRAG**: Knowledge graph construction and querying
- **Tools**: Tool integrations (Mermaid, Gherkin)

### Application Services
Orchestration services that coordinate domain services:
- **Auth**: User authentication and authorization

## Benefits

1. **Maintainability**: Clear separation makes it easy to locate and modify code
2. **Testability**: Layers can be tested independently with mocks
3. **Scalability**: New features can be added without affecting existing code
4. **Reusability**: Domain services can be reused across different applications
5. **Clarity**: Architecture clearly shows the system's structure and dependencies

## Migration Notes

The project was reorganized from a flat structure to this layered architecture:
- Old `services/` → Split into `domain/`, `application/`, and `infrastructure/`
- Old `models/` → Moved to `core/models/`
- Old `clients/` → Moved to `infrastructure/llm/`
- Old `ui/` → Renamed to `presentation/`
- Old `utils/mermaid_renderer.py` → Moved to `utils/renderers/mermaid.py`
- Old `prompts/` → Moved to `domain/prompts/templates/`

All imports have been updated to reflect the new structure.

