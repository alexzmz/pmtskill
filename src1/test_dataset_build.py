from android_world import registry

task_registry = registry.TaskRegistry()

tasks = task_registry.get_registry(family="android_world")

print(len(tasks))

for t in tasks:
    print(t)
