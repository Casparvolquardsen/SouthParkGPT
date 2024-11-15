from openai import OpenAI
from tqdm import tqdm

invalid_lines = {
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Probably.csv': [
        256,
        254,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Toilet_Paper.csv': [
        101
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Faith_Hilling.csv': [
        103,
        166,
        94,
        95,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Miss_Teacher_Bangs_a_Boy.csv': [
        12
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Chef_Goes_Nanners.csv': [
        156
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Fourth_Grade.csv': [
        343
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Two_Days_Before_the_Day_After_Tomorrow.csv': [
        80
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Cr%C3%A8me_Fraiche.csv': [
        145,
        236,
        85,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Pinewood_Derby.csv': [
        235
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/PC_Principal_Final_Justice.csv': [
        67
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Pandemic_2:_The_Startling.csv': [
        243
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Spontaneous_Combustion.csv': [
        381
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/The_Return_of_the_Fellowship_of_the_Ring_to_the_Two_Towers.csv': [
        262
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Taming_Strange.csv': [
        186,
        179,
        180,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Quintuplets_2000.csv': [
        316
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Freak_Strike.csv': [
        149
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Moss_Piglets.csv': [
        6
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Butterballs.csv': [
        163,
        12,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Medicinal_Fried_Chicken.csv': [
        175
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/The_Death_Camp_of_Tolerance.csv': [
        230
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Canada_on_Strike.csv': [
        19,
        20,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/The_Hobbit.csv': [
        134,
        228,
        70,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/W.T.F..csv': [
        284
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Freemium_Isn%27t_Free.csv': [
        136
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/World_War_Zimmerman.csv': [
        53
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Rainforest_Shmainforest.csv': [
        241
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Turd_Burglars.csv': [
        173
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/The_Last_of_the_Meheecans.csv': [
        174,
        175,
        208,
        177,
        118,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Tweek_vs._Craig.csv': [
        64,
        356,
        267,
        179,
        158,
        63,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Two_Guys_Naked_in_a_Hot_Tub.csv': [
        10
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Titties_and_Dragons.csv': [
        212,
        211,
        124,
        237,
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Cock_Magic.csv': [
        255
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/Tsst.csv': [
        249
    ],
    '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned/You_Have_0_Friends.csv': [
        232
    ],
}


prompt_instruction = '''
The following line is from a script of the TV show South Park.
The line contains non-english characters or even non-english text.
Please remove the non-english characters text and replace them with the correct english characters or english translation.
Just respond with the corrected line and do not change the rest of the script.

Here is the line:
'''

client = OpenAI()

for file_path, lines in tqdm(invalid_lines.items()):
    with open(file_path, 'r') as file:
        original_script = file.readlines()

    for line_number in lines:
        line = original_script[line_number]

        prompt = prompt_instruction + line

        response = client.chat.completions.create(
            model="gpt-4o",  # Use the appropriate model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )

        corrected_line = '\n' + response.choices[0].message.content + '\n'
        original_script[line_number] = corrected_line

    with open(file_path, 'w') as file:
        file.writelines(original_script)
