# CCTV Surveillance System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)]()

A full-stack CCTV surveillance application that supports real-time webcam feeds, detection (face, object, license plate, violence, etc.), image/video uploads, customer plans, wanted-face/plate alerts, history tracking, and admin/customer user roles.

---

## Table of Contents

1. [Features](#features)  
2. [Architecture](#architecture)  
3. [Tech Stack](#tech-stack)  
4. [Setup & Installation](#setup--installation)  
   1. [Prerequisites](#prerequisites)  
   2. [Clone & Environment Setup](#clone--environment-setup)  
   3. [Configuration](#configuration)  
   4. [Running the Application](#running-the-application)  
5. [Usage](#usage)  
6. [Database Schema](#database-schema)  
7. [Plans & Permissions](#plans--permissions)  
8. [Detection Types Supported](#detection-types-supported)  
9. [Frontend User Interfaces](#frontend-user-interfaces)  
10. [Contribution Guidelines](#contribution-guidelines)  
11. [Code of Conduct](#code-of-conduct)  
12. [License](#license)  
13. [Contact](#contact)  

---

## Features

- User roles: **Admin** and **Customer**  
  - Admin: add/view customers, manage accounts, view statistics; **cannot** access camera feeds or private detection content.  
  - Customer: access real-time camera/webcam feeds, upload image/video for detection, view detection history, change password, manage profile.  
- Plan-based limits (e.g. Basic, Standard, Premium) → controls how many cameras, features accessible, etc.  
- Real-time webcam feed using WebRTC / similar technology.  
- Detection on images/videos/uploads + in real time: faces, objects, persons, pets, accidents, violence, gun detection, sound alerts (gunshots, screams), license plate detection, helmet detection, vehicle color recognition, speed detection, no-parking zone monitoring, thermal imaging (if supported).  
- Wanted face / wanted license plate: upload wanted items, preview, and match.  
- Annotated outputs + matching text/messages.  
- Detection history: searchable, filterable, with stored data (images, dates, types, etc.).  

---

## Architecture

A high level view of how components interact:

```

\[ Frontend (Web) ] <--> \[ Backend API (Flask / Streamlit) ] <---> \[ Detection Models / Microservices ]
\|                     |                               &#x20;
\|                     |                                 -> \[ Storage (images/videos) ]
\|                     |
\|                     -> \[ Authentication & Authorization ]
|
-> WebRTC / streaming for real-time feed

````

- **Authentication & Authorization** handled via backend, role-based access.  
- **MongoDB** (or similar NoSQL DB) for user data, detection history, wanted items, etc.  
- Separate storage for:  
  - Admin/customer accounts  
  - Detection history + metadata + media files (images/videos)  
- Models for detection either bundled or via separate microservices / modules.

---

## Tech Stack

| Component | Technology / Framework |
|---|---|
| Backend framework | Fast API (or Streamlit for some dashboards) |
| Database | MongoDB |
| Real-time feed | WebRTC or similar streaming mechanism |
| Detection / ML Models | DeepFace, OpenCV, pretrained models for object detection, custom sound event detection etc. |
| Frontend | HTML / CSS / JavaScript (lightweight), templates or React/Vue if extended |
| Storage | Local filesystem or cloud storage (S3 or equivalent) for media outputs |
| Authentication | JWT or session tokens, password hashing etc. |

---

## Setup & Installation

### Prerequisites

- Python 3.8+  
- pip / virtualenv  
- MongoDB instance running (local or remote)  
- (Optional) GPU / CUDA if using heavy detection models  
- (Optional) Dependencies for sound detection, webcam, etc.

### Clone & Environment Setup

```bash
git clone https://github.com/safe-cam/cctv-surveillance.git
cd cctv-surveillance
````

Create a virtual environment, activate it:

```bash
python3 -m venv venv
# On Windows: venv\Scripts\activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

Create a configuration file (e.g. `.env` or config file) with the following (example) environment variables:

```text
# Database
MONGO_URI=mongodb://localhost:27017/cctv_surveillance

# Detection models paths
MODEL_FACE_PATH=path/to/face_model
MODEL_VEHICLE_PATH=path/to/vehicle_model

# Storage paths
MEDIA_UPLOAD_FOLDER=/path/to/uploads
MEDIA_OUTPUT_FOLDER=/path/to/annotated_outputs

```

### Running the Application

First, ensure MongoDB is running. Then:

```bash
# fast api
uvicorn main:app --reload

# OR, use CLI
fastapi dev main.py

# For serving real-time video / WebRTC, ensure correct webcams / RTSP streams are configured
```

(Optional) For development mode with hot reload, debugging:

```bash
export FLASK_DEBUG=1
```

---

## Usage

* **Register / Login** as admin or customer.
* Admin can add customers, assign plans.
* Customer can connect camera(s), or upload images/videos for detection.
* Use the UI to view real-time streams (if applicable), and see detection annotations.
* View detection history, filter by date, type, etc.
* Manage “wanted” faces / plates: upload images, description; system tries matching if detection triggers.

---

## Database Schema (Collections / Models)

Here’s a suggested schema if using MongoDB:

* `users`

  ```json
  {
    _id: ObjectId,
    username: string,
    email: string,
    password_hash: string,
    role: "admin" | "customer",
    profile_photo_url: string,
    plan: "basic" | "standard" | "premium",
    camera_limit: number,
    created_at: datetime,
    updated_at: datetime
  }
  ```

* `cameras`

  ```json
  {
    _id: ObjectId,
    user_id: ObjectId, 
    camera_name: string,
    stream_url / device_info: string,
    is_active: boolean,
    created_at: datetime
  }
  ```

* `detection_history`

  ```json
  {
    _id: ObjectId,
    user_id: ObjectId,
    camera_id: ObjectId (optional, for real-time feed),
    upload_type: "image" | "video" | "webcam",
    detection_type: string,  // e.g. “face”, “accident”, etc.
    result_annotation_path: string,  // where the output image/video with boxes etc is stored
    message: string, // e.g. “Wanted face matched”, etc.
    timestamp: datetime,
    raw_input_path: string,  // original file or stream snapshot if needed
  }
  ```

* `wanted_faces`

  ```json
  {
    _id: ObjectId,
    description: string,
    image_path: string,
    user_uploaded_by: ObjectId,
    created_at: datetime
  }
  ```

* `wanted_plates`

  ```json
  {
    _id: ObjectId,
    plate_number: string,
    image_path: string,
    description: string,
    user_uploaded_by: ObjectId,
    created_at: datetime
  }
  ```

* `plans` (optional, if dynamic)

  ```json
  {
    _id: ObjectId,
    name: "basic" | "standard" | "premium",
    camera_limit: integer,
    features: [ string ],  // list of detection types enabled
    price: number
  }
  ```

---

## Plans & Permissions

| Plan     | Max Cameras | Upload + Detection | Real-Time Feed | Additional Detection Features                 |
| -------- | ----------- | ------------------ | -------------- | --------------------------------------------- |
| Basic    | \~5         | ✔                  | ✔              | Limited / only core features                  |
| Standard | \~10        | ✔                  | ✔              | More detection types, faster processing       |
| Premium  | \~20        | ✔                  | ✔              | All detection types, more resource allocation |

Access control:

* Only customers with an active plan get certain features.
* Admin cannot access user private detections.
* Users can only view their own cameras & detection history.

---

## Detection Types Supported

* Face detection / recognition / wanted-face matching
* Person / object detection
* Pet detection
* Gun detection + sound detection (gunshots, screams)
* Accident detection
* License plate detection & recognition
* Helmet detection
* Vehicle color recognition
* Speed detection
* No-parking / restricted zone monitoring
* Thermal imaging (optional / advanced)

---

## Frontend User Interfaces

* **Login / Register** pages
* **Dashboard** (customer): list of cameras, status, plan info
* **Real-time view**: stream from camera/webcam
* **Upload interface**: upload image or video, select detection type, submit, receive annotated output
* **Detection history page**: preview thumbnails, details, filtering
* **Wanted items page**: for uploading/viewing wanted faces/plates
* **Admin dashboard**: list / manage customers, assign plans, view system-wide statistics (optional)

---

## Contribution Guidelines

We welcome contributions! 🤝

1. Fork the repository.
2. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/my-feature
   ```
3. Make your changes, add tests if relevant.
4. Ensure code follows existing style/format (PEP8, lint).
5. Update or add documentation (README, commenting).
6. Commit your changes with descriptive message.
7. Push your branch:

   ```bash
   git push origin feature/my-feature
   ```
8. Open a Pull Request (PR) against the `main` branch, describe:

   * What you changed
   * Why the change is needed
   * How to test / steps to reproduce
9. Project maintainers will review; make requested changes if any.

---

## Code of Conduct

Please follow these community standards:

* Be respectful in comments / code reviews.
* Write clear, readable code and documentation.
* No discriminatory, harassing or offensive content.
* Adhere to licensing.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contact

Project maintained by **Safe-Cam** organization. For questions, issues, or feature requests:

* Open an [Issue](https://github.com/safe-cam/cctv-surveillance/issues)
<!-- * For direct contact: *your email / team email here* -->
