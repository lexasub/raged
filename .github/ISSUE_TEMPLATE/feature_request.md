---
name: Feature Request
description: Suggest an idea for this project
title: "[FEATURE] "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a feature!
        
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: What problem are you trying to solve?
      placeholder: I'm always frustrated when...
    validations:
      required: true
      
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: Describe the solution you'd like
      placeholder: It would be great if...
    validations:
      required: true
      
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: What alternatives have you considered?
      placeholder: I've also thought about...
      
  - type: textarea
    id: context
    attributes:
      label: Additional Context
      description: Any other context, examples, or mockups
      
  - type: dropdown
    id: language
    attributes:
      label: Language Support
      description: Which programming languages would this feature support?
      multiple: true
      options:
        - Java
        - C++
        - Rust
        - Python
        - TypeScript
        - Other (specify in description)
        
  - type: checkboxes
    id: contribution
    attributes:
      label: Contribution
      description: Would you like to contribute this feature?
      options:
        - label: I'd be willing to implement this feature with guidance
        - label: I can help with testing this feature
