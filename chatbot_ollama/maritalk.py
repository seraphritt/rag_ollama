import maritalk

client = maritalk.MariTalkLocal()

# Iniciando o servidor com uma chave de licença especificada. O executável será baixado em ~/bin/maritalk

client.start_server(license='JTJUJ-TECUL-CRRYW-AWSWV')

response = client.generate_chat([{"role": "user", "content": "Sugira três nomes para o meu peixe"}])

print(response)