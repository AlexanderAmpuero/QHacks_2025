if scanner:
    tracker = BodyTracker()

    running, timer_active, get_pos, start_time = True, False, True, None
    stop_tracking = st.button("Stop Tracking")

    # WebRTC video streamer
    webrtc_streamer(key="example", video_transformer_factory=lambda: tracker)

    while running:
        if stop_tracking:
            st.header("Please wait a moment, PABLO is analyzing your amazing performance!")
            tottime = time.time() - stime
            running = False
            feedback = ai.get_feedback({
                "head_score": head_score / tottime,
                "hand_score": hand_score / tottime,
                "body_score": body_score / tottime,
                "total_time": tottime
            })
            lottie_penguin = load_lottiefile("lottiefiles/penguin.json")
            if lottie_penguin:
                st.markdown('<div class="penguin-container">', unsafe_allow_html=True)
                st_lottie(lottie_penguin, speed=1, loop=False, quality="low", height=200, width=1000, key="penguin_home")
                st.markdown('</div>', unsafe_allow_html=True)

    tracker.release()

    st.write(feedback)
