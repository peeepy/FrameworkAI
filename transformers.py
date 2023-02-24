import requests
import argparse
import json
import os
import io
import discord
from PIL import Image
from pathlib import Path
import re
import base64
import asyncio
from discord.ext import commands
from distutils.util import strtobool
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Create argument parser
parser = argparse.ArgumentParser(description="My Discord Bot")
parser.add_argument(
    "--discord-bot-token", type=str, required=True, help="Discord bot token"
)
parser.add_argument("--channel-id", type=str, required=True, help="Discord channel ID")
parser.add_argument("--save-chats", type=bool, default=False, help="Save chats")
parser.add_argument(
    "--send-greeting", type=bool, default=True, help="Send greeting message"
)
parser.add_argument(
    "--instaboot-mode", type=bool, default=False, help="Enable instaboot mode"
)
parser.add_argument(
    "--instaboot-character", type=int, default=0, help="Instaboot character"
)
parser.add_argument(
    "--instaboot-settings", type=int, default=0, help="Instaboot settings"
)
parser.add_argument("--tts", type=bool, default=False, help="Enable text-to-speech")

# Parse arguments
args = parser.parse_args()
SAVE_CHATS = args.save_chats
SEND_GREETING = args.send_greeting
INSTABOOT_MODE = args.instaboot_mode
DISCORD_BOT_TOKEN = args.discord_bot_token
CHANNEL_ID = args.channel_id
INSTABOOT_CHARACTER = args.instaboot_character
INSTABOOT_SETTINGS = args.instaboot_settings
TTS = args.tts

if not INSTABOOT_MODE:
    INSTABOOT_CHARACTER = 0
    INSTABOOT_SETTINGS = 0
if CHANNEL_ID is not None:
    CHANNEL_ID = CHANNEL_ID.split(",")  # split the string by comma to create an array
else:
    CHANNEL_ID = []  # set the default value to an empty array


async def replace_user_mentions(content, bot):
    user_ids = re.findall(r"<@(\d+)>", content)
    for user_id in user_ids:
        user = await bot.fetch_user(int(user_id))
        if user:
            display_name = user.display_name
            content = content.replace(f"<@{user_id}>", display_name)
    return content


def keyword_scan(response):
    # Define a dictionary with keywords and their corresponding gifs
    json_file = "CharacterInfo/" + char_name + "_gifs.json"
    if os.path.exists(json_file):
        with open(json_file) as f:
            keywords = json.load(f)
        # Iterate through the keywords and check if any of them are in the response_text
        for keyword in keywords:
            if keyword in str(response).lower():
                gif_url = keywords[keyword]
                return gif_url
        # If no keywords are found, return None
        return None
    else:
        return


def dm_scan(message):
    # Define a dictionary with keywords and their corresponding gifs
    if "dm me" in str(message.content).lower():
        return True
    else:
        return False


def upload_character(json_file, img, tavern=False):
    json_file = json_file if type(json_file) == str else json_file.decode("utf-8")
    data = json.loads(json_file)
    outfile_name = data["char_name"]
    i = 1
    while Path(f"Characters/{outfile_name}.json").exists():
        outfile_name = f'{data["char_name"]}_{i:03d}'
        i += 1
    if tavern:
        outfile_name = f"TavernAI-{outfile_name}"
    with open(Path(f"Characters/{outfile_name}.json"), "w") as f:
        f.write(json_file)
    if img is not None:
        img = Image.open(io.BytesIO(img))
        img.save(Path(f"Characters/{outfile_name}.png"))
    print(f'New character saved to "characters/{outfile_name}.json".')
    return outfile_name


def upload_tavern_character(img, name1, name2):
    _img = Image.open(io.BytesIO(img))
    _img.getexif()
    decoded_string = base64.b64decode(_img.info["chara"])
    _json = json.loads(decoded_string)
    _json = {
        "char_name": _json["name"],
        "char_persona": _json["description"],
        "char_greeting": _json["first_mes"],
        "example_dialogue": _json["mes_example"],
        "world_scenario": _json["scenario"],
    }
    _json["example_dialogue"] = (
        _json["example_dialogue"]
        .replace("<USER>", name1)
        .replace("<BOT>", _json["char_name"])
    )
    return upload_character(json.dumps(_json), img, tavern=True)


def generate_response(input_ids):
    output = model.generate(
        input_ids,
        max_length=2048,
        do_sample=True,
        use_cache=True,
        min_new_tokens=10,
        temperature=1.0,
        repetition_penalty=1.01,
        top_p=0.9,
    )
    if output is not None:
        return output
    else:
        return "This is an empty message. Something went wrong. Please check your code!"

def parse_text(generated_text):
    text_lines = [line.strip() for line in str(generated_text).split("\n")]
    aurora_line = next((line for line in reversed(text_lines) if 'AuroraAI' in line), None)
    if aurora_line is not None:
        aurora_line = aurora_line.replace('AuroraAI:', '').strip()
        for i in range(len(text_lines)):
            text_lines[i] = text_lines[i].replace('AuroraAI:', '')
    return aurora_line

characters_folder = 'Characters'
cards_folder = 'Cards'
settings_folder = 'Settings'
characters = []
settings = []
# Check the Cards folder for cards and convert them to characters
try:
    for filename in os.listdir(cards_folder):
        if filename.endswith('.png'):
            with open(os.path.join(cards_folder, filename), 'rb') as read_file:
                img = read_file.read()
                name1 = 'User'
                name2 = 'Character'
                tavern_character_data = upload_tavern_character(img, name1, name2)
            with open(os.path.join(characters_folder, tavern_character_data + '.json')) as read_file:
                character_data = json.load(read_file)
                characters.append(character_data)
            read_file.close()
            os.rename(os.path.join(cards_folder, filename), os.path.join(cards_folder, 'Converted', filename))
except:
    pass
# Load character data from JSON files in the character folder
for filename in os.listdir(characters_folder):
    if filename.endswith('.json'):
        with open(os.path.join(characters_folder, filename)) as read_file:
            character_data = json.load(read_file)
            # Check if there is a corresponding image file for the character
            image_file_jpg = f"{os.path.splitext(filename)[0]}.jpg"
            image_file_png = f"{os.path.splitext(filename)[0]}.png"
            if os.path.exists(os.path.join(characters_folder, image_file_jpg)):
                character_data['char_image'] = image_file_jpg
            elif os.path.exists(os.path.join(characters_folder, image_file_png)):
                character_data['char_image'] = image_file_png
            characters.append(character_data)
# Print a list of characters and let the user choose one
if not INSTABOOT_MODE:
    for i, character in enumerate(characters):
        print(f"{i+1}. {character['char_name']}")
    selected_char = int(input("Please select a character: ")) - 1
    data = characters[selected_char]
else:
    insta_character = int(INSTABOOT_CHARACTER) - 1
    data = characters[insta_character]
char_name = data["char_name"]
char_greeting = data["char_greeting"]
char_dialogue = data["example_dialogue"]
char_dialogue = char_dialogue.replace('"', "")
char_image = data.get("char_image")
# Get the settings for the generation.
for filename in os.listdir(settings_folder):
    if filename.endswith('.settings'):
        with open(os.path.join(settings_folder, filename)) as read_file:
            trimed_name = filename.replace('.settings', '')
            generation_settings = json.load(read_file)
            generation_settings['setting_name'] = trimed_name
            generation_settings['use_story'] = generation_settings.get('use_story', False)
            generation_settings['use_memory'] = generation_settings.get('use_memory', False)
            generation_settings['use_authors_note'] = generation_settings.get('use_authors_note', False)
            generation_settings['use_world_info'] = generation_settings.get('use_world_info', False)
            generation_settings['max_context_length'] = generation_settings.get('max_length', 1818)
            generation_settings['max_length'] = generation_settings.get('genamt', 60)
            generation_settings['rep_pen'] = generation_settings.get('rep_pen', 1.01)
            generation_settings['rep_pen_range'] = generation_settings.get('rep_pen_range', 1024)
            generation_settings['rep_pen_slope'] = generation_settings.get('rep_pen_slope', 0.9)
            generation_settings['temperature'] = generation_settings.get('temperature', 1)
            generation_settings['tfs'] = generation_settings.get('tfs', 0.9)
            generation_settings['top_a'] = generation_settings.get('top_a', 0)
            generation_settings['top_k'] = generation_settings.get('top_k', 40)
            generation_settings['top_p'] = generation_settings.get('top_p', 0.9)
            generation_settings['typical'] = generation_settings.get('typical', 1)
            generation_settings['frmttriminc'] = generation_settings.get('frmttriminc', False)
            generation_settings['frmtrmblln'] = generation_settings.get('frmtrmblln', True)
            generation_settings['sampler_order'] = generation_settings.get('sampler_order', [6, 0, 1, 2, 3, 4, 5])
            settings.append(generation_settings)
if not (INSTABOOT_MODE):
    for i, setting in enumerate(settings):
        print(f"{i+1}. {generation_settings['setting_name']}")
    selected_settings= int(input("Please select a settings file: ")) - 1
    setting = settings[selected_settings]
elif (INSTABOOT_MODE):
    insta_setting = int(INSTABOOT_SETTINGS) - 1
    setting = settings[insta_setting]
num_lines_to_keep = 20
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='/', intents=intents)
core_prompt = f"{char_name}'s Persona: {data['char_persona']}\n" + f"Scenario: {data['world_scenario']}\n"
conversation_history =  f'Example Dialogue: \n' + \
                        f'{char_dialogue}\n' + \
                        f'<START>\n' + \
                        f'{char_name}: {char_greeting}\n'
dm_history = conversation_history
starting_dialogue = char_greeting.replace("{", "").replace("}", "").replace("'", "")

# load model
print("\u001b[0;45mLoading model...\u001b[0;0m")

model_path = "Model"
model = "pygmalion-6b"
revision = "dev"
model_name = f"{model}-{revision}"
model_dir = os.path.join(model_path, model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    "PygmalionAI/pygmalion-6b", revision=revision, torch_dtype=torch.float16
).to(device)
tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")
print("\u001b[0;45mPygmalion 6B loaded.\u001b[0;0m")

generate = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

@bot.event
async def on_ready():
    try:
        with open(f"Characters/{char_image}", "rb") as f:
            avatar_data = f.read()
        await bot.user.edit(username=char_name, avatar=avatar_data)
    except FileNotFoundError:
        with open(f"Characters/default.png", "rb") as f:
            avatar_data = f.read()
        await bot.user.edit(username=char_name, avatar=avatar_data)
        print(f"No image found for {char_name}. Setting image to default.")
    except discord.errors.HTTPException as error:
        if (
            error.code == 50035
            and "Too many users have this username, please try another" in error.text
        ):
            await bot.user.edit(username=char_name + "BOT", avatar=avatar_data)
        elif (
            error.code == 50035
            and "You are changing your username or Discord Tag too fast. Try again later."
            in error.text
        ):
            pass
        else:
            raise error
    print(f"{bot.user} has connected to Discord!")
    if SEND_GREETING:
        for channel_id in CHANNEL_ID:
            channel = bot.get_channel(int(channel_id))
            if not os.path.exists(f"Logs/{char_name}_{channel_id}_chat_logs.txt"):
                await channel.send(char_greeting)
                with open(f"Logs/{char_name}_{channel_id}_chat_logs.txt", "a") as file:
                    file.write(f"{char_name}:{char_greeting}\n")
            elif not SAVE_CHATS:
                await channel.send(char_greeting)
            else:
                return


@bot.event
async def on_message(message):
    global conversation_history
    global dm_history
    world_info = ""
    if message.author == bot.user:
        return
    name_str = str(message.author.name)
    path_to_dm_log = f"Logs/{char_name}_{name_str}_chat_logs.txt"
    path_to_channel_log = f"Logs/{char_name}_{message.channel.id}_chat_logs.txt"
    dm_message = (
        f"A DM? What things do you wish to discuss with {char_name} in private?"
    )
    message_content = await replace_user_mentions(message.content, bot)
    # DM Handling
    if isinstance(message.channel, discord.DMChannel):
        if not os.path.exists(path_to_dm_log):
            dm_history += (
                f"{name_str}:{message_content}\n" + f"{char_name}:{dm_message}\n"
            )
            await bot.get_cog("log_cog").save_history(path_to_dm_log, dm_history)
            await message.reply(dm_message)
            return
        else:
            world_info_check = await bot.get_cog("scan_message").world_info(
                message_content, data
            )
            if not (world_info_check == "error"):
                world_info = world_info_check
            if message.attachments and message.attachments[0].filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".gif")
            ):
                # The message has an attached image, pass it to the imagecaption cog
                image_response = await bot.get_cog("image_caption").image_comment(
                    message, message_content
                )
                print(image_response)
                await bot.get_cog("log_cog").save_logs(
                    path_to_dm_log, name_str, image_response
                )
                if os.path.exists(path_to_dm_log):
                    with open(path_to_dm_log, "r") as f:
                        lines = f.readlines()[-15:]
                        dm_history = "".join(lines)
            user = await replace_user_mentions(message.author.name, bot)
            prompt = (
                core_prompt + conversation_history + f"{user}: {message_content}\n{char_name}:"
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            output = generate_response(input_ids)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            aurora_line = parse_text(generated_text)
            print(prompt)
            print(generated_text)

        if image_response:
            await bot.get_cog("image_caption").image_comment(
                message, message_content
            )
            if image_response:
                dm_history += f"{name_str}:{image_response}\n" + f"{char_name}:{response}\n"
            await bot.get_cog("log_cog").save_logs(path_to_dm_log, char_name, response)
            gif_url = await bot.get_cog("scan_message").gif_scan(aurora_line, data)
            await message.channel.send(aurora_line)
            if gif_url:
                await message.channel.send(str(gif_url))
            else:
                await bot.get_cog("log_cog").save_logs(
                    path_to_dm_log, name_str, message_content
                )
                if os.path.exists(path_to_dm_log):
                    with open(path_to_dm_log, "r") as f:
                        lines = f.readlines()[-15:]
                        dm_history = "".join(lines)
                print(f"{name_str}:{message_content}")
                user = await replace_user_mentions(message.author.name, bot)
                prompt = (
                    core_prompt + conversation_history + f"{user}: {message_content}\n{char_name}:"
                )
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                output = generate_response(input_ids)
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                aurora_line = parse_text(generated_text)
                print(prompt)
                print(generated_text)
                dm_history += (
                    f"{name_str}:{message_content}\n" + f"{char_name}:{aurora_line}\n"
                )
                await bot.get_cog("log_cog").save_logs(
                    path_to_dm_log, char_name, response
                )
                gif_url = await bot.get_cog("scan_message").gif_scan(aurora_line, data)
                await message.channel.send(aurora_line)
                if gif_url:
                    await message.channel.send(str(gif_url))
    # Channel Message Handling
    elif message.content.startswith("."):
        return
    else:
        for channel_id in CHANNEL_ID:
            if (isinstance(message.author, discord.Member)) and (
                message.author.nick is not None
            ):
                name = message.author.nick
            else:
                name = message.author.name
            name_str = str(name)
            channel = bot.get_channel(int(channel_id))
            if message.channel == channel:
                world_info_check = await bot.get_cog("scan_message").world_info(
                    f"{name}:{message_content}", data
                )
                if not (world_info_check == "error"):
                    world_info = world_info_check
                if message.author == bot.user:
                    return
                if message.attachments and message.attachments[
                    0
                ].filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                    # The message has an attached image, pass it to the imagecaption cog
                    image_response = await bot.get_cog("image_caption").image_comment(
                        message, message_content
                    )
                    print(image_response)
                    await bot.get_cog("log_cog").save_logs(
                        path_to_channel_log, name_str, message_content
                    )
                    if os.path.exists(path_to_channel_log):
                        with open(path_to_channel_log, "r") as f:
                            lines = f.readlines()[-15:]
                            conversation_history = "".join(lines)
                    user = await replace_user_mentions(message.author.name, bot)
                    prompt = (
                        core_prompt + conversation_history + f"{user}: {message_content}\n{char_name}:"
                    )
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    output = generate_response(input_ids)
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    print(prompt)
                    print(generated_text)
                    conversation_history += (
                        f"{name_str}:{image_response}\n" + f"{char_name}:{aurora_line}\n"
                    )
                    await bot.get_cog("log_cog").save_logs(
                        path_to_channel_log, char_name, aurora_line
                    )
                    gif_url = await bot.get_cog("scan_message").gif_scan(aurora_line, data)
                    await message.channel.send(aurora_line)
                    if gif_url:
                        await message.channel.send(str(gif_url))
                else:
                    await bot.get_cog("log_cog").save_history(
                        path_to_channel_log, conversation_history
                    )
                    await bot.get_cog("log_cog").save_logs(
                        path_to_channel_log, name_str, message_content
                    )
                    if os.path.exists(path_to_channel_log):
                        with open(path_to_channel_log, "r") as f:
                            lines = f.readlines()[-15:]
                            conversation_history = "".join(lines)
                    print(f"{name_str}:{message_content}")
                    if bot.user in message.mentions and 'dm' in message_content.lower():
                        dm_scanned = await bot.get_cog("scan_message").dm_scan(message)
                        user = await replace_user_mentions(message.author.name, bot)
                        request_dm = f"Yes, {user}? You wanted to speak privately?"
                        if dm_scanned:
                            user = message.author
                            channel = await user.create_dm()
                            dm_history += f"{char_name}:{request_dm}\n"
                            await channel.send(request_dm)
                            await bot.get_cog("log_cog").save_logs(
                                path_to_channel_log, char_name, request_dm
                            )
                            return
                    user = await replace_user_mentions(message.author.name, bot)
                    prompt = (
                        core_prompt + conversation_history + f"{user}: {message_content}\n{char_name}:"
                    )
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    output = generate_response(input_ids)
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    aurora_line = parse_text(generated_text)
                    print(generated_text)
                    conversation_history += (
                        f"{name_str}:{message_content}\n" + f"{char_name}:{aurora_line}\n"
                    )
                    await bot.get_cog("log_cog").save_logs(
                        path_to_channel_log, char_name, aurora_line
                    )
                    gif_url = await bot.get_cog("scan_message").gif_scan(aurora_line, data)
                    await message.reply(aurora_line)
                    if not gif_url is None:
                        await message.reply(str(gif_url))
                    return
                
                    if message_content == ("aurora.retry"):
                        botmessage = get_channel(CHANNEL_ID).history(limit=2).flatten()[1]
                        retrymessage = get_channel(CHANNEL_ID).history(limit=2).flatten()[0]
                        if message.author.bot:
                            print("Regenerating.")
                            await botmessage.delete()
                            await usermessage.delete()
                            user = await replace_user_mentions(message.author.name, bot)
                            prompt = (
                            core_prompt + conversation_history + f"{user}: {message_content}\n{char_name}:"
                            )   
                            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                            output = generate_response(input_ids)
                            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                            aurora_line = parse_text(generated_text)
                            print(prompt)

async def load_cogs() -> None:
    for file in os.listdir(f"{os.path.realpath(os.path.dirname(__file__))}/cogs"):
        if file.endswith(".py"):
            extension = file[:-3]
            try:
                await bot.load_extension(f"cogs.{extension}")
            except Exception as e:
                exception = f"{type(e).__name__}: {e}"
                print(exception)


asyncio.run(load_cogs())
bot.run(DISCORD_BOT_TOKEN)