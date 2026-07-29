#!/bin/bash

echo "🚀 بدء نشر مشروع كشف سرطان الجلد..."

# تثبيت المتطلبات
echo "📦 تثبيت المكتبات..."
pip install -r backend/requirements.txt

# تشغيل الـ API
echo "🔥 تشغيل الـ API على المنفذ 8000..."
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload