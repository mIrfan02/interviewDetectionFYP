-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 06, 2024 at 08:13 AM
-- Server version: 10.4.27-MariaDB
-- PHP Version: 8.0.25

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `fyp`
--

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `id` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `email` varchar(120) NOT NULL,
  `password` varchar(60) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`id`, `username`, `email`, `password`) VALUES
(1, 'ismail', 'ismail@gmail.com', '123456'),
(2, 'Muhammad ismail', 'm.ismail110022@gmail.com', 'kfdjkfd'),
(3, 'talha ', 'talha@gmail.com', '12345'),
(4, 'atta', 'atta@gmail.com', '12345'),
(5, 'ali', 'ali@gmail.com', '123'),
(7, 'ismail khan', 'ismail11@gmail.com', '12345'),
(9, 'alikhan', 'alikhan5684@gmail.com', '12345');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `email` varchar(120) NOT NULL,
  `password` varchar(50) NOT NULL,
  `history` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`history`))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `email`, `password`, `history`) VALUES
(1, 'ali', 'ali@gmail.com', '123', '{\r\n    \"Date\": [\r\n        {\r\n            \"feedback\": \"Feedback 1\",\r\n            \"fluency\": \"Fluency 1\",\r\n            \"voice-emotions\": \"Voice Emotions 1\",\r\n            \"faceEmotion\": \"Face Emotion 1\",\r\n            \"summarization\": \"Summarization 1\"\r\n        },\r\n        {\r\n            \"feedback\": \"Feedback 2\",\r\n            \"fluency\": \"Fluency 2\",\r\n            \"voice-emotions\": \"Voice Emotions 2\",\r\n            \"faceEmotion\": \"Face Emotion 2\",\r\n            \"summarization\": \"Summarization 2\"\r\n        }\r\n    ]\r\n}\r\n');

-- --------------------------------------------------------

--
-- Table structure for table `usersall`
--

CREATE TABLE `usersall` (
  `id` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `email` varchar(120) NOT NULL,
  `password` varchar(60) NOT NULL,
  `history` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`history`))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `usersall`
--

INSERT INTO `usersall` (`id`, `username`, `email`, `password`, `history`) VALUES
(1, 'ali', 'ali@gmail.com', '123', '{\"2024-05-04\": [{\"feedback\": \"The interviewee\'s emotions fluctuated throughout the interview. Their vocal emotions ranged from angry to happy to fearful, indicating a mix of negative and positive feelings. Their facial expressions were predominantly negative, with frequent switches between sadness and surprise. This suggests that the interviewee may have been feeling overwhelmed or uncomfortable during the interview. To improve, the interviewee should try to maintain a positive and confident demeanor, both verbally and nonverbally. They should articulate their thoughts clearly and concisely, and make an effort to appear engaged and interested in the conversation. Additionally, they should be prepared to answer questions about their skills and experience, and be able to articulate their strengths and weaknesses.\", \"fluency\": \"The transcription appears to be disjointed and incomplete, with the interviewee\'s thoughts and sentences trailing off abruptly. The lack of continuity and coherence suggests that the interviewee may have been confused or experiencing difficulty expressing themselves clearly. The use of fragmented phrases and incomplete sentences indicates that the interviewee may have been struggling to find the right words or organize their thoughts effectively. This disruption in the flow of speech could be a sign of anxiety, nervousness, or a lack of preparation. The interviewee may benefit from additional guidance and support in developing a more coherent and organized speaking style.\", \"voic_emotions\": [\"Angry\", \"Happy\", \"Fear\"], \"face_emotions\": [\"happy\", \"sad\", \"surprise\", \"sad\", \"surprise\", \"sad\", \"surprise\", \"sad\", \"surprise\", \"sad\"]}]}'),
(2, 'ismail', 'ismail@gmail.com', '123', '{\"2024-05-04\": [{\"feedback\": \"This interviewee should perform a little better to impress the interviewer. The shifting emotions captured by both voice and face recognition suggest fluctuations in the interviewee\'s composure and confidence levels. The initial \'Angry\' voice emotion indicates a possible underlying anxiety or frustration, which may have contributed to the \'sad\' face emotion in the second interval. While the subsequent \'Happy\' voice emotion and \'surprise\' face emotions show moments of engagement, the recurring \'sad\' face emotions point to a lack of consistent enthusiasm and may be interpreted as a sign of disinterest or nervousness. To improve their performance, the interviewee should focus on managing their emotions by practicing deep breathing exercises, maintaining eye contact, and projecting a positive and confident demeanor throughout the interview.\", \"fluency\": \"The interviewee seems to be confused and lacking fluency in the transcription provided. The sentences are disjointed, with missing words and incomplete thoughts. The interviewee frequently hesitates, stumbles over words, and loses their train of thought. The overall impression is one of nervousness and disorganization. Additionally, the interviewee\'s responses often lack detail and substance. They provide general statements without providing specific examples or elaborations. This suggests that the interviewee may not be fully prepared or knowledgeable about the topic being discussed.\", \"voic_emotions\": [\"Angry\", \"Happy\", \"Angry\"], \"face_emotions\": [\"happy\", \"sad\", \"surprise\", \"sad\", \"surprise\", \"sad\", \"surprise\", \"sad\", \"surprise\", \"sad\"], \"summary\": \"in speech delivery emotions help the speaker to make the audience understand the message the main reason for giving a speech is to get the messages across the earliest.\"}]}'),
(3, 'irfan', 'irfan@gmail.com', '123', '{\"2024-05-06\": [{\"feedback\": \"The interviewee displayed a range of emotions throughout the interview, including anger, happiness, and fear in their voice, and happiness, sadness, and surprise in their facial expressions. While the interviewee\'s emotional expressiveness could indicate engagement and authenticity, the frequent shifts in emotions may have also conveyed nervousness or discomfort. To improve, the interviewee could focus on maintaining a consistent and professional demeanor, while still allowing for appropriate emotional responses to the interview questions. This could involve practicing self-regulation techniques, such as deep breathing or visualization, to manage emotional fluctuations and project a sense of composure and confidence.\", \"fluency\": \"The transcription reveals that the interviewee is somewhat confused and lacks fluency. The speech is disjointed, with frequent pauses and incomplete sentences. The interviewee struggles to organize their thoughts and ideas, which results in a lack of coherence and flow. The use of filler words such as \\\"um\\\" and \\\"like\\\" further indicates hesitation and uncertainty. Overall, the transcription suggests that the interviewee is not relaxed and is struggling to articulate their thoughts effectively. Feedback should focus on providing guidance on improving fluency, organization, and clarity in speech delivery.\", \"voic_emotions\": [\"Angry\", \"Happy\", \"Fear\"], \"face_emotions\": [\"happy\", \"sad\", \"surprise\", \"sad\", \"surprise\", \"sad\", \"surprise\", \"sad\", \"surprise\", \"sad\"], \"summary\": \"in speech delivery emotions help the speaker to make the audience understand the message the main reason for giving a speech is to get the messages across the earliest.\"}]}');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `usersall`
--
ALTER TABLE `usersall`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `user`
--
ALTER TABLE `user`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;

--
-- AUTO_INCREMENT for table `usersall`
--
ALTER TABLE `usersall`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
