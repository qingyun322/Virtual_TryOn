from random import randint
import os
def random_person(config):
    database = config['database']
    person_folder = os.path.join(database, 'person')
    person_names = os.listdir(person_folder)
    idx = randint(0, len(person_names) - 1)
    person_name = person_names[idx]
    return person_name

def random_cloth(config):
    database = config['database']
    cloth_folder = os.path.join(database, 'cloth')
    cloth_names = os.listdir(cloth_folder)
    idx = randint(0, len(cloth_names) - 1)
    cloth_name = cloth_names[idx]
    return cloth_name
