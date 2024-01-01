import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

import net

MODEL = None


def predict_tonality(text):
    tonality_prediction = net.parse_tonality(net.predict(MODEL, text))
    return tonality_prediction


# Define the /start command handler
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Welcome! Send me any text, and I will predict its tonality.')


async def handle_text(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text
    tonality_prediction = predict_tonality(user_text)

    await update.message.reply_text(f'Tonality Prediction: {tonality_prediction}')


# Main function to run the bot
def main() -> None:
    global MODEL
    MODEL = net.load()

    logging.basicConfig(level='INFO')
    app = Application.builder().token('6464278470:AAGC6VsAomTKStzcV3hlJe7uRfCOSX3DYXU').build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Start the Bot
    app.run_polling()


if __name__ == '__main__':
    main()
