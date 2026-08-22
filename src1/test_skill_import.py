from pmtskill_v2.skills.store import SkillStore
from dataclasses import fields

store = SkillStore("/home/zmz/Workspace/gui/src1/runtime/skill_library.sqlite3")

skills = [
    skill
    for skill in store.list_skills()
    if skill.metadata.get("approved_for_planning", True)
]

print("count:", len(skills))
# for skill in skills:
#     print("=" * 80)
#     print("name:", skill.name)
#     print("kind:", skill.kind)
#     # print("body:", skill.body)
#     # print("metadata", skill.metadata)
#     print("topology:", skill.topology)
#     # break


# for skill in skills:
#     store.rollback_raw_skill_compile(skill.skill_id)
