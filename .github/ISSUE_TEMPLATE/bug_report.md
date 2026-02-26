---
name: Bug Report
description: Report a bug or unexpected behavior
title: "[BUG] "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
        
  - type: textarea
    id: description
    attributes:
      label: Description
      description: Briefly describe the issue
      placeholder: What happened? What did you expect to happen?
    validations:
      required: true
      
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Detailed steps to reproduce the issue
      placeholder: |
        1. Index project with `ast-rag init .`
        2. Run `ast-rag query "..."`
        3. See error...
    validations:
      required: true
      
  - type: input
    id: environment-python
    attributes:
      label: Python Version
      placeholder: "3.11"
    validations:
      required: true
      
  - type: input
    id: environment-neo4j
    attributes:
      label: Neo4j Version
      placeholder: "5.15"
      
  - type: input
    id: environment-qdrant
    attributes:
      label: Qdrant Version
      placeholder: "1.14.0"
      
  - type: textarea
    id: logs
    attributes:
      label: Relevant Logs
      description: Please copy and paste any relevant log output
      render: shell
      
  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
