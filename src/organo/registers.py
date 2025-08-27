from organo.registries.module_registry import ModuleRegistry

model_registry = ModuleRegistry("model")
criterion_registry = ModuleRegistry("criterion")
task_registry = ModuleRegistry("task")
optimizer_registry = ModuleRegistry("optimizer")
logger_registry = ModuleRegistry("logger")
