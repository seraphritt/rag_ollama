import ollama

stream = ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'translate it to English: 主动爆料的徐蓝鹤万万没想到会听到这个消息，他下意识问道，“还是之前那个？”这话说的没头没脑的，不过观众还是都听懂了。 把句子补全就是，“你还是向之前那个告白了？” 陆惊渡淡淡反问道，“不然呢。” “！” 徐蓝鹤差点直接爆粗。 他来就是打算来爆个大料的，没想到最大的料还是陆惊渡自己爆的。 徐蓝鹤有些不满，“那你不早点跟我说？”他们还是不是好兄弟了？ \
陆惊渡表情平静，但眼底却有笑意流露，“昨晚太忙了，没来得及。” \
徐蓝鹤在内心忍不住啧啧啧了几声，瞧瞧这眼角眉梢的春风，鬼都知道昨晚肯定发生什么好事了。 \
他真心实意地笑了下，“恭喜啊。”好兄弟终于得偿所愿了。 \
“谢谢。” \
观众这时候也反应过来了。 \
【所以陆惊渡是向咸鱼王告白的吗？】 \
【这两人算不算是正式官宣了？】 \
【昨晚居然有告白，我们观众怎么不知道？】 \
【有什么是我们尊贵的vip观众不能看的？】 \
导演愣了好几秒之后，才用手指了下陆惊渡和宁小渔，“你们？在一起了？” \
陆惊渡点点头，“对。” \
导演哈哈一笑，“恭喜恭喜啊。” \
其他人也纷纷送上祝福。 \
 可能是被当事人本人爆了大料，所以后面的几个连线，都没观众仔细听。 \
直到轮到节目组连线沈约约时，观众才勉强打起精神来。  \
刚才影帝的朋友爆了一个大料，那咸鱼王的朋友呢？ 不爆个大料可说不过去了。 \
结果一连线，沈约约反倒成了最积极吃瓜的那个人，“什么？你们竟然在一起了？什么时候的事？我怎么不知道？”'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
