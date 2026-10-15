Fix ``assign_params`` binding positional-only parameters from a namespace,
using defaults for optional gaps while preserving later partial calls.
These names now bind positionally even when ``**kwargs`` accepts the same name.
Assignments that skip a required positional-only parameter before a later
assignment now raise ``TypeError`` when the partial is constructed.
