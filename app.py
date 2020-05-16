"""
Module to run the bot.
Telegram API Key (API_KEY), greeting message (greeting), and caption for photo (caption) must be  declared in the config.py

Author: Igor Belov (igooor.bb@gmail.com)
"""

import config
import logging
import image
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters) 

def start(update, context):
    """
    The callback function for the 'start' command.
    Sends greeting message.
    """
    update.message.reply_text(config.greeting)

def photo(update, context):
    """
    The callback function tracking incoming photos.
    Generates an image and sends it to the user.
    """
    context.bot.send_message(chat_id=update.effective_message.chat_id, text=config.caption)
    photo_bytes = update.message.photo[-1].get_file().download_as_bytearray()
    
    msg = context.bot.send_photo(chat_id=update.effective_message.chat_id, photo=image.generate_image(photo_bytes))

def sticker(update, context):
    """
    The callback function tracking incoming stickers.
    Generates an image and sends it to the user.
    """
    if (not update.message.sticker.is_animated):
        context.bot.send_message(chat_id=update.effective_message.chat_id, text=config.caption)
        sticker_bytes = update.message.sticker.get_file().download_as_bytearray()

        msg = context.bot.send_photo(chat_id=update.effective_message.chat_id, photo=image.generate_image(sticker_bytes))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    updater = Updater(config.API_KEY, use_context=True)
    
    dp = updater.dispatcher

    start_handler = CommandHandler(['start', 'help'], start)
    dp.add_handler(start_handler)


    photo_handler = MessageHandler(Filters.photo, photo)
    dp.add_handler(photo_handler)

    sticker_handler = MessageHandler(Filters.sticker, sticker)
    dp.add_handler(sticker_handler)

    updater.start_polling()
    updater.idle()
