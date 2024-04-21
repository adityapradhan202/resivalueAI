import discord
from discord.ext import commands

from joblib import load
poly_converter = load('fpc.joblib')
mlmodel = load('fmodel.joblib')

intents = discord.Intents().all()
bot = commands.Bot(command_prefix="!", intents=intents, case_insensitive=True)

bot.remove_command("help")

@bot.command()
async def help(ctx):
    await ctx.reply(f"To predict house price type this command (!predict a b c d) where a is the number of rooms other than bedrooms, b is the total area in sqft, c is the number of bathrooms and d is the number of bedrooms :D")

@bot.command()
async def about(ctx):
    await ctx.reply('resivalueAI is a discord bot powered my AI and Machine Learning which gives you the best prediction on house prices, based on the dataset on which it is being trained :D')


@bot.command()
async def predict(ctx, nrooms:int, total_areasqft:int, nbathrooms:int, nbedrooms:int):

    user_input = [[nrooms, total_areasqft, nbathrooms, nbedrooms]]
    user_input_polyfeature = poly_converter.fit_transform(user_input)
    
    predicted_price = mlmodel.predict(user_input_polyfeature)
    output = round(predicted_price[0],2)

    await ctx.reply(f"The price will be nearly {output} US dollars")

bot.run('YOUR_API_KEY_HERE')