import random
import json
import os
import time
import math
import datetime
import requests
from universal_funcs import get_request_backend, send_message, delete_message, storage_bucket, send_photo, send_chat_action, react_to_message, forward_message, number_to_emojis, wait_open, get_media_content, get_bot_token, send_animation, normalize_lang
import telepot
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
from Publisher import POSTMAIL_CHAT_LINK
from Server import NUMBER_CHATS
import Distortioner
import Fortune
from loc import i18n, VALID_LANGS

bloblist_ideiadesenho = list(storage_bucket.list_blobs(prefix="IdeiaDesenho"))
bloblist_death = list(storage_bucket.list_blobs(prefix="Death"))
bloblist_bff = list(storage_bucket.list_blobs(prefix="Countdown/BFF"))
bloblist_patas = list(storage_bucket.list_blobs(prefix="Countdown/Patas"))
bloblist_fursmeet = list(storage_bucket.list_blobs(prefix="Countdown/FurSMeet"))
bloblist_furcamp = list(storage_bucket.list_blobs(prefix="Countdown/Furcamp"))
bloblist_pawstral = list(storage_bucket.list_blobs(prefix="Countdown/Pawstral"))
custom_commands = list(dict.fromkeys([folder.name.split('/')[1] for folder in storage_bucket.list_blobs(prefix="Custom/")]))
NEW_CHAT_LINK = "https://t.me/CookieMWbot?startgroup=new"
WEBSITE_LINK = "https://cookiebotfur.net"
TEST_CHAT_LINK = "https://t.me/+mX6W3tGXPew2OTIx"
UPDATES_CHANNEL_LINK = "https://t.me/cookiebotupdates"

# Initialize with corpus
fortunes = {i18n._lang_chain(lang): Fortune.FortuneGenerator(i18n.get_file("sorte.txt", lang=lang).splitlines()) for lang in VALID_LANGS}

def decapitalize(s, upper_rest = False):
    return ''.join([s[:1].lower(), (s[1:].upper() if upper_rest else s[1:])])

def pv_default_message(cookiebot, msg, chat_id, is_alternate_bot):
    send_chat_action(cookiebot, chat_id, 'typing')
    language = normalize_lang(msg.get('from', {}).get('language_code'))

    bot_key_map = {1: "bombot", 2: "pawsy", 3: "tarinbot", 4: "connectbot"}
    bot_key = bot_key_map.get(is_alternate_bot, "cookiebot")

    first_name = msg.get('from', {}).get('first_name', '') or i18n.get("pv.user_fallback", lang=language)
    num_chats_emoji = number_to_emojis(NUMBER_CHATS)

    name = i18n.get(f"bots.{bot_key}.name", lang=language)
    title = i18n.get("pv.title", lang=language, name=name, first_name=first_name)
    desc = i18n.get(f"bots.{bot_key}.description", lang=language, num_chats=num_chats_emoji)
    addl = i18n.get(f"bots.{bot_key}.additional_info", lang=language, default="")
    cmds = i18n.get("pv.commands", lang=language, default="")
    contact = i18n.get("pv.contact", lang=language)

    message = f"{title}\n\n{desc}{addl}\n{cmds}{contact}"

    reply_markup = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=i18n.get("pv.buttons.add_group", lang=language), url=NEW_CHAT_LINK)],
        [InlineKeyboardButton(text=i18n.get("pv.buttons.website", lang=language), url=WEBSITE_LINK)],
        [InlineKeyboardButton(text=i18n.get("pv.buttons.shared_posts", lang=language), url=POSTMAIL_CHAT_LINK)],
        [InlineKeyboardButton(text=i18n.get("pv.buttons.updates", lang=language), url=UPDATES_CHANNEL_LINK)],
        [InlineKeyboardButton(text=i18n.get("pv.buttons.test_group", lang=language), url=TEST_CHAT_LINK)]
    ])
    send_message(cookiebot, chat_id, message, reply_markup=reply_markup)

def privacy_statement(cookiebot, msg, chat_id, language):
    send_chat_action(cookiebot, chat_id, 'typing')
    text = i18n.get("privacy", lang=language)
    send_message(cookiebot, chat_id, text, msg_to_reply=msg, parse_mode='HTML')

def is_alive(cookiebot, msg, chat_id, language, is_alternate_bot=0):
    react_to_message(msg, '👍', is_alternate_bot=is_alternate_bot)
    send_chat_action(cookiebot, chat_id, 'typing')
    text = i18n.get("alive", lang=language)
    send_message(cookiebot, chat_id, text + str(datetime.datetime.now()), msg)

def analyze(cookiebot, msg, chat_id, language, is_alternate_bot=0):
    send_chat_action(cookiebot, chat_id, 'typing')
    if not 'reply_to_message' in msg:
        text = i18n.get("analyze", lang=language)
        send_message(cookiebot, chat_id, text, msg)
        return
    react_to_message(msg, '🤔', is_alternate_bot=is_alternate_bot)
    result = ''
    for item in msg['reply_to_message']:
        result += str(item) + ': ' + str(msg['reply_to_message'][item]) + '\n'
    send_message(cookiebot, chat_id, result, msg_to_reply=msg)

def list_groups(cookiebot, chat_id):
    send_chat_action(cookiebot, chat_id, 'typing')
    file_path = 'list_groups.json'
    existing_chats, new_chats, removed_chats = [], [], []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf8') as file:
            try:
                existing_chats = json.load(file)
            except json.JSONDecodeError:
                pass
    groups = get_request_backend('registers')
    for group in groups:
        try:
            chat = cookiebot.getChat(int(group['id']))
            time.sleep(0.4)
            if 'title' in chat and chat['type'] in ['group', 'supergroup']:
                chat_info = f"{group['id']} - {chat['title']}"
                if not any(chat.startswith(chat_info.split()[0]) for chat in existing_chats):
                    new_chats.append(chat_info)
                cookiebot.sendMessage(chat_id, chat_info)
                time.sleep(0.1)
        except Exception:
            removed_chats.append(group['id'])
    cookiebot.sendMessage(chat_id, i18n.get("groups.total", value = len(groups) - len(removed_chats)))
    with open(file_path, 'w', encoding='utf8') as file:
        json.dump([f"{group['id']} - {group.get('title', '')}" for group in groups], file)
    if new_chats:
        cookiebot.sendMessage(chat_id,  i18n.get("groups.new", text = ', '.join(new_chats)))
    if removed_chats:
        cookiebot.sendMessage(chat_id,  i18n.get("groups.remove", text = ', '.join(removed_chats)))

def broadcast_message(cookiebot, msg):
    groups = get_request_backend('registers')
    for group in groups:
        try:
            group_id = group['id']
            send_message(cookiebot, int(group_id), msg['text'].replace('/broadcast', ''))
            time.sleep(0.5)
        except Exception:
            pass

def list_commands(cookiebot, msg, chat_id, language):
    send_chat_action(cookiebot, chat_id, 'typing')
    string = i18n.get_file("Cookiebot_functions.txt", lang=language)
    send_message(cookiebot, chat_id, string, msg_to_reply=msg)

def notify_fun_off(cookiebot, msg, chat_id, language):
    text = i18n.get("fun_off", lang=language)
    send_message(cookiebot, chat_id, text, msg)

def notify_utility_off(cookiebot, msg, chat_id, language):
    text = i18n.get("utility_off", lang=language)
    send_message(cookiebot, chat_id, text, msg)

def drawing_idea(cookiebot, msg, chat_id, language):
    send_chat_action(cookiebot, chat_id, 'upload_photo')
    idea_id = random.randint(0, len(bloblist_ideiadesenho)-1)
    blob = bloblist_ideiadesenho[idea_id]
    photo = blob.generate_signed_url(datetime.timedelta(minutes=15), method='GET')
    caption = i18n.get("drawing_idea", lang=language, idea_id=idea_id)
    send_photo(cookiebot, chat_id, photo, caption=caption, msg_to_reply=msg)

def custom_command(cookiebot, msg, chat_id, language):
    send_chat_action(cookiebot, chat_id, 'upload_photo')
    bloblist = list(storage_bucket.list_blobs(prefix="Custom/"+msg['text'].replace('/', '').replace("@CookieMWbot", '').replace("@pawstralbot", '').split()[0]))
    if len(msg['text'].split()) > 1 and msg['text'].split()[1].isdigit():
        image_id = int(msg['text'].split()[1])
    else:
        image_id = random.randint(0, len(bloblist)-1)
    ctx = {
        "name": msg['text'].replace('/', '').replace('@CookieMWbot', '').split()[0].capitalize(),
        "image_id": image_id
    }
    photo = bloblist[image_id].generate_signed_url(datetime.timedelta(minutes=15), method='GET')
    caption = i18n.get("custom_photo", lang=language, **ctx)
    send_photo(cookiebot, chat_id, photo, msg_to_reply=msg, caption=caption)

def roll_dice(cookiebot, msg, chat_id, language):
    send_chat_action(cookiebot, chat_id, 'typing')
    start = msg['text'].split(" ")[0]
    if start in ("/dado","/dice"):
        text = i18n.get("dice_exemple", lang=language)
        send_message(cookiebot, chat_id, text)
    else:
        if len(msg['text'].split()) == 1:
            vezes = 1
        else:
            vezes = int(msg['text'].replace("@CookieMWbot", '').replace("@pawstralbot", '').split()[1])
            vezes = max(min(20, vezes), 1)
        limite = int(msg['text'].replace("@CookieMWbot", '').replace("@pawstralbot", '').split()[0][2:])
        resposta = f"(d{limite}) "
        if vezes == 1:
            resposta += f"🎲 -> {random.randint(1, limite)}"
        else:
            for vez in range(vezes):
                ctx = {
                    "vez": vez+1,
                    "roll": random.randint(1, limite)
                }
                resposta += i18n.get("dice_roll", lang=language, **ctx)
        send_message(cookiebot, chat_id, resposta, msg_to_reply=msg)

def age(cookiebot, msg, chat_id, language):
    if not " " in msg['text']:
        text = i18n.get("age", lang=language)
        send_message(cookiebot, chat_id, text, msg, language)
    else:
        nome = msg['text'].replace("/idade ", '').replace("/edad ", '').replace("/age ", '').replace("/idade@CookieMWbot", '').replace("/age@CookieMWbot", '').replace("/edad@CookieMWbot", '').split()[0]
        response = json.loads(requests.get(f"https://api.agify.io?name={nome}", timeout=10).text)
        registered_times = response['count']
        if registered_times == 0:
            text = i18n.get("not_know", lang=language)
            send_message(cookiebot, chat_id, text, msg)
        else:
            ctx = {
                    "age": response['age'],
                    "registered_times": registered_times
                }
            text = i18n.get("age_yes", lang=language, **ctx)
            send_message(cookiebot, chat_id, text, msg)

def gender(cookiebot, msg, chat_id, language):
    send_chat_action(cookiebot, chat_id, 'typing')
    if not " " in msg['text']:
        text = i18n.get("gender_exemple", lang=language)
        send_message(cookiebot, chat_id, text, msg)
    else:
        nome = msg['text'].replace("/genero ", '').replace("/gênero ", '').replace("/gender ", '').replace("/genero@CookieMWbot", '').replace("/gênero@CookieMWbot", '').replace("/gender@CookieMWbot", '').split()[0]
        response = json.loads(requests.get(f"https://api.genderize.io?name={nome}", timeout=10).text)
        genero = response['gender']
        probability = response['probability']
        registered_times = response['count']
        if registered_times == 0:
            text = i18n.get("not_know", lang=language)
            send_message(cookiebot, chat_id, text, msg)
        else:
            ctx = {
                "probability": probability*100,
                "registered_times": registered_times
            }
            text = i18n.get(f"gender.{genero}", lang=language, **ctx)
            send_message(cookiebot, chat_id, text, msg)

def firecracker(cookiebot, msg, chat_id, thread_id=None, is_alternate_bot=0):
    react_to_message(msg, '🎉', is_alternate_bot=is_alternate_bot)
    send_message(cookiebot, chat_id, "fiiiiiiii.... ", msg_to_reply=msg)
    time.sleep(0.1)
    amount = random.randint(5, 20)
    while amount > 0:
        if random.choice([True, False]):
            n = random.randint(1, amount)
        else:
            n = 1
        send_message(cookiebot, chat_id, "pra "*n, thread_id=thread_id, is_alternate_bot=is_alternate_bot)
        amount -= n
    send_message(cookiebot, chat_id, "<b> 💥POOOOOOOWW💥 </b>", thread_id=thread_id, is_alternate_bot=is_alternate_bot)

def complaint(cookiebot, msg, chat_id, language):
    send_chat_action(cookiebot, chat_id, 'upload_photo')
    caption=  i18n.get("complaint", lang=language, user= msg['from']['first_name'])
    if language == 'pt':
        with open('Static/reclamacao/milton_pt.jpg', 'rb') as photo:
            send_photo(cookiebot, chat_id, photo, caption=caption, msg_to_reply=msg)
    else:
        with open('Static/reclamacao/milton_eng.jpg', 'rb') as photo:
            send_photo(cookiebot, chat_id, photo, caption=caption, msg_to_reply=msg)

def complaint_answer(cookiebot, msg, chat_id, language):
    delete_message(cookiebot, telepot.message_identifier(msg['reply_to_message']))
    send_chat_action(cookiebot, chat_id, 'upload_audio')
    protocol = f"{random.randint(10, 99)}-{random.randint(100000, 999999)}/{datetime.datetime.now().year}"
    with open(f"Static/reclamacao/{random.choice([file for file in os.listdir('Static/reclamacao') if file.endswith('.wav')])}", 'rb') as hold_audio:
        hold_msg = cookiebot.sendVoice(chat_id, hold_audio, caption=f"Protocol: {protocol}", reply_to_message_id=msg['message_id'])
    time.sleep(random.randint(10, 20))
    delete_message(cookiebot, telepot.message_identifier(hold_msg))
    answer = i18n.get_random_line("answers.txt", lang=language)
    send_message(cookiebot, chat_id, answer, msg_to_reply=msg)

def event_countdown(cookiebot, msg, chat_id, language, is_alternate_bot):
    react_to_message(msg, '🔥', is_alternate_bot=is_alternate_bot)
    send_chat_action(cookiebot, chat_id, 'upload_photo')
    if msg['text'].lower().startswith('/patas'):
        day, month, year = 11, 12, 2026
        calltoaction = random.choice(i18n.get("event.patas.cta", lang=language))
        pic = bloblist_patas[random.randint(0, len(bloblist_patas)-1)].generate_signed_url(datetime.timedelta(minutes=15), method='GET')
        daysremaining = (datetime.datetime(year, month, day) - datetime.datetime.now()).days + 1
        if -5 <= daysremaining <= 0:
            caption = "https://www.youtube.com/watch?v=JsOVJ1PAC6s&ab_channel=TheVibeGuide"
        else:
            while daysremaining < -5:
                daysremaining += 365
            caption = f"<b> Faltam {number_to_emojis(daysremaining)} dias para o Patas! </b>\n\n<i> {calltoaction} </i>\n🐾🍌🐾🐒🐾🍌🐾🐒🐾🍌🐾🐒🐾🍌\n\n📆 {day} a {day+3}/{month}, Sorocaba Park Hotel\n💻 Ingressos em: patas.site\n📲 Grupo do evento: @EventoPatas"
    elif msg['text'].lower().startswith('/bff'):
        day, month, year = 17, 7, 2026
        calltoaction = random.choice(i18n.get("event.bff.cta", lang=language))
        pic = bloblist_bff[random.randint(0, len(bloblist_bff)-1)].generate_signed_url(datetime.timedelta(minutes=15), method='GET')
        daysremaining = (datetime.datetime(year, month, day) - datetime.datetime.now()).days + 1
        if -5 <= daysremaining <= 0:
            caption = "https://www.youtube.com/watch?v=JsOVJ1PAC6s&ab_channel=TheVibeGuide"
        else:
            while daysremaining < -5:
                daysremaining += 365
            caption = f"<b> Faltam {number_to_emojis(daysremaining)} dias para a Brasil FurFest 2026 - Sem Tempo Irmão! </b>\n\n<i> {calltoaction} </i>\n🐾🟩🐾🟨🐾🟩🐾🟨🐾🟩🐾🟨🐾🟩\n\n📆 {day} a {day+2}/{month}\n📍 Hotel Premium - Campinas\n💻 Site: brasilfurfest.com.br, upgrades até 1 mês antes do evento através do email reg@brasilfurfest.com.br\n📲 Grupo do evento: @brasilfurfest"
    elif msg['text'].lower().startswith('/fursmeet'):
        day, month, year = 21, 11, 2025
        calltoaction = random.choice(i18n.get("event.fursmeet.cta", lang=language))
        pic = bloblist_fursmeet[random.randint(0, len(bloblist_fursmeet)-1)].generate_signed_url(datetime.timedelta(minutes=15), method='GET')
        daysremaining = (datetime.datetime(year, month, day) - datetime.datetime.now()).days + 1
        if -5 <= daysremaining <= 0:
            caption = "https://www.youtube.com/watch?v=JsOVJ1PAC6s&ab_channel=TheVibeGuide"
        else:
            while daysremaining < -5:
                daysremaining += 365
            caption = f"<b> Faltam {number_to_emojis(daysremaining)} dias para o FurSMeet {year}! </b>\n\n<i> {calltoaction} </i>\n🦕🦖🦫🦕🦖🦫🦕🦖🦫🦕🦖🦫🦕🦖🦫\n\n📆 {day} a {day+2}/{month}, Santa Maria, Rio Grande do Sul\n🎫Link para comprar ingresso: fursmeet.carrd.co\n💻 Informações no site: fursmeet.wixsite.com/fursmeet\n📲 Grupo do evento: @fursmeetchat"
    elif msg['text'].lower().startswith('/furcamp'):
        day, month, year = 13, 2, 2026
        calltoaction = random.choice(i18n.get("event.furcamp.cta", lang=language))
        pic = bloblist_furcamp[random.randint(0, len(bloblist_furcamp)-1)].generate_signed_url(datetime.timedelta(minutes=15), method='GET')
        daysremaining = (datetime.datetime(year, month, day) - datetime.datetime.now()).days + 1
        if -5 <= daysremaining <= 0:
            caption = "https://www.youtube.com/watch?v=JsOVJ1PAC6s&ab_channel=TheVibeGuide"
        else:
            while daysremaining < -5:
                daysremaining += 365
            caption = f"<b> Faltam {number_to_emojis(daysremaining)} dias para o FurCamp! </b>\n\n<i> {calltoaction} </i>\n🐾🌲🐾🌳🐾🌲🐾🌳🐾🌲🐾🌳\n\n📆 {day} a {day+4}/{month}, Acampamento Terra do Saber - Cajamar - SP\n💻 Ingressos em: furcamp.com\n📲 Grupo do evento: @FurcampOficial"
    elif msg['text'].lower().startswith('/pawstral'):
        day, month, year = 29, 8, 2025
        calltoaction = random.choice(i18n.get("event.pawstral.cta", lang=language))
        pic = bloblist_pawstral[random.randint(0, len(bloblist_pawstral)-1)].generate_signed_url(datetime.timedelta(minutes=15), method='GET')
        daysremaining = (datetime.datetime(year, month, day) - datetime.datetime.now()).days + 1
        if -5 <= daysremaining <= 0:
            caption = "https://www.youtube.com/watch?v=JsOVJ1PAC6s&ab_channel=TheVibeGuide"
        else:
            while daysremaining < -5:
                daysremaining += 365
            caption = f"<b> {number_to_emojis(daysremaining)} days left until Pawstral! </b>\n\n<i> {calltoaction} </i>\n🇨🇱⭐🐈🇨🇱⭐🐈🇨🇱⭐🐈🇨🇱⭐🐈🇨🇱⭐🐈\n\n📆 {day} a {day+2}/{month}, Santiago de Chile\n💻 Tickets at: https://pawstral.cl/\n📲 Event chat: @PawstralFurcon"
    else:
        text = random.choice(i18n.get("event.error", lang=language))
        send_message(cookiebot, chat_id, text, msg)
        return
    send_photo(cookiebot, chat_id, pic, caption=caption, msg_to_reply=msg, is_alternate_bot=is_alternate_bot)

def unearth(cookiebot, msg, chat_id, thread_id=None):
    send_chat_action(cookiebot, chat_id, 'typing')
    for _ in range(100):
        try:
            chosenid = random.randint(1, msg['message_id'])
            forward_message(cookiebot, chat_id, chat_id, chosenid, thread_id=thread_id)
            return chosenid
        except Exception:
            return None

def death(cookiebot, msg, chat_id, language):
    react_to_message(msg, '👻')
    send_chat_action(cookiebot, chat_id, 'upload_photo')
    fileblob = bloblist_death[random.randint(0, len(bloblist_death)-1)]
    filename = fileblob.name
    fileurl = fileblob.generate_signed_url(datetime.timedelta(minutes=15), method='GET')
    if len(msg['text'].split()) > 1:
        caption = '💀💀💀 ' + msg['text'].split()[1]
    elif 'reply_to_message' in msg:
        caption = '💀💀💀 ' + msg['reply_to_message']['from']['first_name']
    else:
        caption = '💀💀💀 ' + '@'+msg['from']['username'] if 'username' in msg['from'] else msg['from']['first_name']
    
    variants = i18n.get("death.variants", lang=language)
    template = i18n.get("death.template", lang=language, variant= random.choice(variants))
    caption += template
    line = i18n.get_random_line("death.txt", lang=language)
    additional = i18n.get("death.Reason", lang=language, line= line)
    caption += additional
    if filename.endswith('.gif'):
        send_animation(cookiebot, chat_id, fileurl, caption=caption, msg_to_reply=msg)
    else:
        send_photo(cookiebot, chat_id, fileurl, caption=caption, msg_to_reply=msg)

def fortune_cookie(cookiebot, msg, chat_id, language):
    send_chat_action(cookiebot, chat_id, 'upload_photo')
    anim_id = send_animation(cookiebot, chat_id, 'https://s12.gifyu.com/images/S5e9b.gif', msg_to_reply=msg)
    line = fortunes.get(i18n._lang_chain(language)).generate()
    numbers = []
    tens = []
    while len(numbers) < 6:
        number = random.randint(1, 99)
        if math.floor(number / 10) not in tens:
            numbers.append(number)
            tens.append(math.floor(number / 10))
    numbers_str = ' '.join([str(number) for number in numbers])
    answer = i18n.get("luck", lang=language, line=line, num=numbers_str)
    time.sleep(3)
    delete_message(cookiebot, (str(chat_id), str(anim_id)))
    send_chat_action(cookiebot, chat_id, 'typing')
    send_message(cookiebot, chat_id, answer, msg_to_reply=msg, parse_mode='HTML')

def destroy(cookiebot, msg, chat_id, language, is_alternate_bot=0):
    instru = i18n.get("destroy.instru", lang=language)
    if msg['text'].endswith('pfp'):
        send_chat_action(cookiebot, chat_id, 'upload_photo')
        token = get_bot_token(is_alternate_bot)
        file_path_telegram = cookiebot.getFile(cookiebot.getUserProfilePhotos(msg['from']['id'])['photos'][0][-1]['file_id'])['file_path']
        r = requests.get(f"https://api.telegram.org/file/bot{token}/{file_path_telegram}", allow_redirects=True, timeout=10)
        filename = file_path_telegram.split('/')[-1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        Distortioner.distortioner(filename)
        with open('distorted.jpg', 'rb') as photo:
            cookiebot.sendPhoto(chat_id, photo, reply_to_message_id=msg['message_id'])
        os.remove('distorted.jpg')
        os.remove(filename)
    elif not 'reply_to_message' in msg:
        send_message(cookiebot, chat_id, instru, msg)
    elif 'video' in msg['reply_to_message']:
        text = i18n.get("destroy.video", lang=language)
        send_message(cookiebot, chat_id, text, msg)
    elif 'photo' in msg['reply_to_message']:
        send_chat_action(cookiebot, chat_id, 'upload_photo')
        photo_file = get_media_content(cookiebot, msg['reply_to_message'], 'photo', is_alternate_bot=is_alternate_bot, downloadfile=True)
        Distortioner.distortioner(photo_file)
        with open('distorted.jpg', 'rb') as photo:
            cookiebot.sendPhoto(chat_id, photo, reply_to_message_id=msg['message_id'])
        os.remove('distorted.jpg')
        os.remove(photo_file)
    elif 'audio' in msg['reply_to_message'] or 'voice' in msg['reply_to_message']:
        send_chat_action(cookiebot, chat_id, 'upload_voice')
        if 'audio' in msg['reply_to_message']:
            audio_file = get_media_content(cookiebot, msg['reply_to_message'], 'audio', is_alternate_bot=is_alternate_bot, downloadfile=True)
        else:
            audio_file = get_media_content(cookiebot, msg['reply_to_message'], 'voice', is_alternate_bot=is_alternate_bot, downloadfile=True)
        Distortioner.distort_audiofile(audio_file, 10, 1, 'distorted.mp3')
        with open('distorted.mp3', 'rb') as audio:
            cookiebot.sendAudio(chat_id, audio, reply_to_message_id=msg['message_id'])
        os.remove('distorted.mp3')
        os.remove(audio_file)
    elif 'sticker' in msg['reply_to_message']:
        if msg['reply_to_message']['sticker']['is_animated'] or msg['reply_to_message']['sticker']['is_video']:
            text = i18n.get("destroy.gif", lang=language)
            send_message(cookiebot, chat_id, text, msg)
            return
        send_chat_action(cookiebot, chat_id, 'upload_photo')
        sticker_file = get_media_content(cookiebot, msg['reply_to_message'], 'sticker', is_alternate_bot=is_alternate_bot, downloadfile=True)
        Distortioner.process_image(sticker_file, 'distorted.png', 25)
        with open('distorted.png', 'rb') as sticker:
            cookiebot.sendSticker(chat_id, sticker, reply_to_message_id=msg['message_id'])
        os.remove('distorted.png')
        os.remove(sticker_file)
    elif 'animation' in msg['reply_to_message']:
        text = i18n.get("destroy.gif", lang=language)
        send_message(cookiebot, chat_id, text, msg)
    else:
        send_message(cookiebot, chat_id, instru, msg)

