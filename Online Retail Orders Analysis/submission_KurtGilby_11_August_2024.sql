/*

-----------------------------------------------------------------------------------------------------------------------------------
                                               Guidelines
-----------------------------------------------------------------------------------------------------------------------------------

The provided document is a guide for the project. Follow the instructions and take the necessary steps to finish
the project in the SQL file			

-----------------------------------------------------------------------------------------------------------------------------------

                                                         Queries
                                               
-----------------------------------------------------------------------------------------------------------------------------------*/
USE orders;
-- 1. WRITE A QUERY TO DISPLAY CUSTOMER FULL NAME WITH THEIR TITLE (MR/MS), BOTH FIRST NAME AND LAST NAME ARE IN UPPER CASE WITH 
-- CUSTOMER EMAIL ID, CUSTOMER CREATIONDATE AND DISPLAY CUSTOMER’S CATEGORY AFTER APPLYING BELOW CATEGORIZATION RULES:
	SELECT 
    CASE
        WHEN
            CUSTOMER_GENDER = 'F'
        THEN
            CONCAT('MS. ',
                    UPPER(CUSTOMER_FNAME),
                    ' ',
                    UPPER(CUSTOMER_LNAME))
        ELSE CONCAT('MR. ',
                UPPER(CUSTOMER_FNAME),
                ' ',
                UPPER(CUSTOMER_LNAME))
    END 'CUSTOMER_FULL_NAME',
    CUSTOMER_EMAIL AS 'CUSTOMER_EMAIL_ID',
    CUSTOMER_CREATION_DATE AS 'CUSTOMER_CREATIONDATE',
    CASE
        WHEN YEAR(CUSTOMER_CREATION_DATE) < 2005 THEN 'CATEGORY A'
        WHEN
            YEAR(CUSTOMER_CREATION_DATE) >= 2005
                AND YEAR(CUSTOMER_CREATION_DATE) < 2011
        THEN
            'CATEGORY B'
        ELSE 'CATEGORY C'
    END AS 'CUSTOMER_CATEGORY'
FROM
    online_customer;

-- 2. WRITE A QUERY TO DISPLAY THE FOLLOWING INFORMATION FOR THE PRODUCTS, WHICH HAVE NOT BEEN SOLD:  PRODUCT_ID, PRODUCT_DESC, 
-- PRODUCT_QUANTITY_AVAIL, PRODUCT_PRICE,INVENTORY VALUES(PRODUCT_QUANTITY_AVAIL*PRODUCT_PRICE), NEW_PRICE AFTER APPLYING DISCOUNT 
-- AS PER BELOW CRITERIA. SORT THE OUTPUT WITH RESPECT TO DECREASING VALUE OF INVENTORY_VALUE.
	SELECT 
    PRODUCT_ID,
    PRODUCT_DESC,
    PRODUCT_QUANTITY_AVAIL,
    PRODUCT_PRICE,
    (PRODUCT_QUANTITY_AVAIL * PRODUCT_PRICE) AS 'INVENTORY_VALUES',
    CASE
        WHEN PRODUCT_PRICE > 20000 THEN PRODUCT_PRICE * (.80)
        WHEN PRODUCT_PRICE > 10000 THEN PRODUCT_PRICE * (.85)
        ELSE PRODUCT_PRICE * (.90)
    END AS 'NEW_PRICE'
FROM
    product
WHERE
    PRODUCT_ID NOT IN (SELECT DISTINCT
            PRODUCT_ID
        FROM
            order_items)
ORDER BY (PRODUCT_QUANTITY_AVAIL * PRODUCT_PRICE) DESC;
    



-- 3. WRITE A QUERY TO DISPLAY PRODUCT_CLASS_CODE, PRODUCT_CLASS_DESCRIPTION, COUNT OF PRODUCT TYPE IN EACH PRODUCT CLASS, 
-- INVENTORY VALUE (P.PRODUCT_QUANTITY_AVAIL*P.PRODUCT_PRICE). INFORMATION SHOULD BE DISPLAYED FOR ONLY THOSE PRODUCT_CLASS_CODE 
-- WHICH HAVE MORE THAN 1,00,000 INVENTORY VALUE. SORT THE OUTPUT WITH RESPECT TO DECREASING VALUE OF INVENTORY_VALUE.
	-- [NOTE: TABLES TO BE USED -PRODUCT, PRODUCT_CLASS]
SELECT 
    pc.PRODUCT_CLASS_CODE,
    pc.PRODUCT_CLASS_DESC,
    COUNT(p.PRODUCT_ID) 'COUNT_PRODUCT_TYPE',
    SUM(p.PRODUCT_QUANTITY_AVAIL * p.PRODUCT_PRICE) AS 'INVENTORY_VALUE'
FROM
    product_class pc
        LEFT JOIN
    product p ON pc.PRODUCT_CLASS_CODE = p.PRODUCT_CLASS_CODE
GROUP BY 1 , 2
HAVING SUM(p.PRODUCT_QUANTITY_AVAIL * p.PRODUCT_PRICE) > 100000
ORDER BY 4 DESC;



-- 4. WRITE A QUERY TO DISPLAY CUSTOMER_ID, FULL NAME, CUSTOMER_EMAIL, CUSTOMER_PHONE AND COUNTRY OF CUSTOMERS WHO HAVE CANCELLED 
-- ALL THE ORDERS PLACED BY THEM(USE SUB-QUERY)
	-- [NOTE: TABLES TO BE USED - ONLINE_CUSTOMER, ADDRESSS, ORDER_HEADER]
SELECT 
    c.CUSTOMER_ID,
    CASE
        WHEN
            c.CUSTOMER_GENDER = 'F'
        THEN
            CONCAT('MS. ',
                    UPPER(c.CUSTOMER_FNAME),
                    ' ',
                    UPPER(c.CUSTOMER_LNAME))
        ELSE CONCAT('MR. ',
                UPPER(c.CUSTOMER_FNAME),
                ' ',
                UPPER(c.CUSTOMER_LNAME))
    END 'CUSTOMER_FULL_NAME',
    c.CUSTOMER_EMAIL,
    c.CUSTOMER_PHONE,
    a.COUNTRY
FROM
    online_customer c
        LEFT JOIN
    address a ON c.ADDRESS_ID = a.ADDRESS_ID
WHERE
    c.CUSTOMER_ID IN (SELECT DISTINCT
            CUSTOMER_ID
        FROM
            order_header
        WHERE
            CUSTOMER_ID IN (SELECT DISTINCT
                    CUSTOMER_ID
                FROM
                    (SELECT 
                        CUSTOMER_ID, COUNT(DISTINCT ORDER_STATUS) ORDER_STATUS_COUNT
                    FROM
                        order_header
                    GROUP BY 1
                    HAVING COUNT(DISTINCT ORDER_STATUS) = 1) ONE_ORDER_STATUS)
                AND ORDER_STATUS = 'Cancelled');

        
-- 5. WRITE A QUERY TO DISPLAY SHIPPER NAME, CITY TO WHICH IT IS CATERING, NUMBER OF CUSTOMER CATERED BY THE SHIPPER IN THE CITY AND 
-- NUMBER OF CONSIGNMENTS DELIVERED TO THAT CITY FOR SHIPPER DHL(9 ROWS)
	-- [NOTE: TABLES TO BE USED -SHIPPER, ONLINE_CUSTOMER, ADDRESSS, ORDER_HEADER]
SELECT 
    s.SHIPPER_NAME,
    a.CITY,
    COUNT(DISTINCT c.CUSTOMER_ID) COUNT_CUSTOMER_CATERED,
    SUM(CASE WHEN o.ORDER_STATUS = 'Shipped' THEN 1 ELSE 0 END) COUNT_CONSIGNMENTS_DELIVERED
FROM
    shipper s
        INNER JOIN
    order_header o ON s.SHIPPER_ID = o.SHIPPER_ID
        AND s.SHIPPER_NAME = 'DHL'
        INNER JOIN
    online_customer c ON o.CUSTOMER_ID = c.CUSTOMER_ID
        INNER JOIN
    address a ON c.ADDRESS_ID = a.ADDRESS_ID
GROUP BY 1 , 2;

-- 6. WRITE A QUERY TO DISPLAY CUSTOMER ID, CUSTOMER FULL NAME, TOTAL QUANTITY AND TOTAL VALUE (QUANTITY*PRICE) SHIPPED WHERE MODE 
-- OF PAYMENT IS CASH AND CUSTOMER LAST NAME STARTS WITH 'G'
	-- [NOTE: TABLES TO BE USED -ONLINE_CUSTOMER, ORDER_ITEMS, PRODUCT, ORDER_HEADER]
SELECT 
    c.CUSTOMER_ID,
    CASE
        WHEN
            c.CUSTOMER_GENDER = 'F'
        THEN
            CONCAT('MS. ',
                    UPPER(c.CUSTOMER_FNAME),
                    ' ',
                    UPPER(c.CUSTOMER_LNAME))
        ELSE CONCAT('MR. ',
                UPPER(c.CUSTOMER_FNAME),
                ' ',
                UPPER(c.CUSTOMER_LNAME))
    END 'CUSTOMER_FULL_NAME',
    SUM(oi.PRODUCT_QUANTITY) AS TOTAL_QUANTITY,
    SUM(oi.PRODUCT_QUANTITY * p.PRODUCT_PRICE) AS TOTAL_VALUE
FROM
    online_customer c
        INNER JOIN
    order_header o ON o.CUSTOMER_ID = c.CUSTOMER_ID
        AND o.ORDER_STATUS = 'Shipped'
        AND o.PAYMENT_MODE = 'Cash'
        AND c.CUSTOMER_LNAME LIKE 'G%'
        INNER JOIN
    order_items oi ON oi.ORDER_ID = o.ORDER_ID
        INNER JOIN
    product p ON oi.PRODUCT_ID = p.PRODUCT_ID
GROUP BY 1 , 2;

-- 7. WRITE A QUERY TO DISPLAY ORDER_ID AND VOLUME OF BIGGEST ORDER (IN TERMS OF VOLUME) THAT CAN FIT IN CARTON ID 10  
	-- [NOTE: TABLES TO BE USED -CARTON, ORDER_ITEMS, PRODUCT]

SELECT 
    oi.ORDER_ID,
    SUM((oi.PRODUCT_QUANTITY) * (p.LEN * p.WIDTH * p.HEIGHT)) AS ORDER_VOLUME
FROM
    order_items oi
        LEFT JOIN
    product p ON oi.PRODUCT_ID = p.PRODUCT_ID
GROUP BY 1
HAVING SUM((oi.PRODUCT_QUANTITY) * (p.LEN * p.WIDTH * p.HEIGHT)) <= (SELECT 
        LEN * WIDTH * HEIGHT AS VOLUME
    FROM
        carton
    WHERE
        CARTON_ID = 10)
ORDER BY 2 DESC
LIMIT 1;

-- 8. WRITE A QUERY TO DISPLAY PRODUCT_ID, PRODUCT_DESC, PRODUCT_QUANTITY_AVAIL, QUANTITY SOLD, AND SHOW INVENTORY STATUS OF 
-- PRODUCTS AS BELOW AS PER BELOW CONDITION:
	-- A.FOR ELECTRONICS AND COMPUTER CATEGORIES, 
		-- i.IF SALES TILL DATE IS ZERO THEN SHOW 'NO SALES IN PAST, GIVE DISCOUNT TO REDUCE INVENTORY',
        -- ii.IF INVENTORY QUANTITY IS LESS THAN 10% OF QUANTITY SOLD, SHOW 'LOW INVENTORY, NEED TO ADD INVENTORY', 
        -- iii.IF INVENTORY QUANTITY IS LESS THAN 50% OF QUANTITY SOLD, SHOW 'MEDIUM INVENTORY, NEED TO ADD SOME INVENTORY', 
        -- iv.IF INVENTORY QUANTITY IS MORE OR EQUAL TO 50% OF QUANTITY SOLD, SHOW 'SUFFICIENT INVENTORY'
	-- B.FOR MOBILES AND WATCHES CATEGORIES, 
		-- i.IF SALES TILL DATE IS ZERO THEN SHOW 'NO SALES IN PAST, GIVE DISCOUNT TO REDUCE INVENTORY', 
        -- ii.IF INVENTORY QUANTITY IS LESS THAN 20% OF QUANTITY SOLD, SHOW 'LOW INVENTORY, NEED TO ADD INVENTORY',  
        -- iii.IF INVENTORY QUANTITY IS LESS THAN 60% OF QUANTITY SOLD, SHOW 'MEDIUM INVENTORY, NEED TO ADD SOME INVENTORY', 
        -- iv.IF INVENTORY QUANTITY IS MORE OR EQUAL TO 60% OF QUANTITY SOLD, SHOW 'SUFFICIENT INVENTORY'
	-- C.REST OF THE CATEGORIES, 
		-- i.IF SALES TILL DATE IS ZERO THEN SHOW 'NO SALES IN PAST, GIVE DISCOUNT TO REDUCE INVENTORY', 
        -- ii.IF INVENTORY QUANTITY IS LESS THAN 30% OF QUANTITY SOLD, SHOW 'LOW INVENTORY, NEED TO ADD INVENTORY',  
        -- iii.IF INVENTORY QUANTITY IS LESS THAN 70% OF QUANTITY SOLD, SHOW 'MEDIUM INVENTORY, NEED TO ADD SOME INVENTORY', 
        -- iv. IF INVENTORY QUANTITY IS MORE OR EQUAL TO 70% OF QUANTITY SOLD, SHOW 'SUFFICIENT INVENTORY'
        
			-- [NOTE: TABLES TO BE USED -PRODUCT, PRODUCT_CLASS, ORDER_ITEMS] (USE SUB-QUERY)
SELECT 
    p.PRODUCT_ID,
    p.PRODUCT_DESC,
    COALESCE(p.PRODUCT_QUANTITY_AVAIL, 0) PRODUCT_QUANTITY_AVAIL,
    COALESCE(sq.QUANTITY_SOLD, 0) QUANTITY_SOLD,
    CASE
        WHEN
            pc.PRODUCT_CLASS_DESC IN ('Electronics' , 'Computers')
        THEN
            CASE
                WHEN COALESCE(sq.QUANTITY_SOLD, 0) = 0 THEN 'NO SALES IN PAST, GIVE DISCOUNT TO REDUCE INVENTORY'
                WHEN COALESCE(p.PRODUCT_QUANTITY_AVAIL, 0) < 0.1 * sq.quantity_sold THEN 'LOW INVENTORY, NEED TO ADD INVENTORY'
                WHEN COALESCE(p.PRODUCT_QUANTITY_AVAIL, 0) < 0.5 * sq.quantity_sold THEN 'MEDIUM INVENTORY, NEED TO ADD SOME INVENTORY'
                ELSE 'SUFFICIENT INVENTORY'
            END
        WHEN
            pc.PRODUCT_CLASS_DESC IN ('Mobiles' , 'Watches')
        THEN
            CASE
                WHEN COALESCE(sq.QUANTITY_SOLD, 0) = 0 THEN 'NO SALES IN PAST, GIVE DISCOUNT TO REDUCE INVENTORY'
                WHEN COALESCE(p.PRODUCT_QUANTITY_AVAIL, 0) < 0.2 * sq.QUANTITY_SOLD THEN 'LOW INVENTORY, NEED TO ADD INVENTORY'
                WHEN COALESCE(p.PRODUCT_QUANTITY_AVAIL, 0) < 0.6 * sq.QUANTITY_SOLD THEN 'MEDIUM INVENTORY, NEED TO ADD SOME INVENTORY'
                ELSE 'SUFFICIENT INVENTORY'
            END
        ELSE CASE
            WHEN COALESCE(sq.QUANTITY_SOLD, 0) = 0 THEN 'NO SALES IN PAST, GIVE DISCOUNT TO REDUCE INVENTORY'
            WHEN COALESCE(p.PRODUCT_QUANTITY_AVAIL, 0) < 0.3 * sq.QUANTITY_SOLD THEN 'LOW INVENTORY, NEED TO ADD INVENTORY'
            WHEN COALESCE(p.PRODUCT_QUANTITY_AVAIL, 0) < 0.7 * sq.QUANTITY_SOLD THEN 'MEDIUM INVENTORY, NEED TO ADD SOME INVENTORY'
            ELSE 'SUFFICIENT INVENTORY'
        END
    END AS INVENTORY_STATUS
FROM
    product p
        LEFT JOIN
    product_class pc ON p.PRODUCT_CLASS_CODE = pc.PRODUCT_CLASS_CODE
        LEFT JOIN
    (SELECT 
        oi.PRODUCT_ID,
            COALESCE(SUM(oi.PRODUCT_QUANTITY), 0) AS QUANTITY_SOLD
    FROM
        order_items oi
    GROUP BY oi.PRODUCT_ID) sq ON p.PRODUCT_ID = sq.PRODUCT_ID;
    

-- 9. WRITE A QUERY TO DISPLAY PRODUCT_ID, PRODUCT_DESC AND TOTAL QUANTITY OF PRODUCTS WHICH ARE SOLD TOGETHER WITH PRODUCT ID 201 
-- AND ARE NOT SHIPPED TO CITY BANGALORE AND NEW DELHI. DISPLAY THE OUTPUT IN DESCENDING ORDER WITH RESPECT TO TOT_QTY.(USE SUB-QUERY)
	-- [NOTE: TABLES TO BE USED -ORDER_ITEMS,PRODUCT,ORDER_HEADER, ONLINE_CUSTOMER, ADDRESS]
    
SELECT 
    p.PRODUCT_ID,
    p.PRODUCT_DESC,
    SUM(oi.PRODUCT_QUANTITY) AS TOTAL_QUANTITY
FROM
    order_items oi
        LEFT JOIN
    product p ON oi.PRODUCT_ID = p.PRODUCT_ID
        LEFT JOIN
    order_header oh ON oi.ORDER_ID = oh.ORDER_ID
        LEFT JOIN
    online_customer oc ON oh.CUSTOMER_ID = oc.CUSTOMER_ID
        LEFT JOIN
    address a ON oc.ADDRESS_ID = a.ADDRESS_ID
WHERE
    oi.ORDER_ID IN (SELECT 
            oi_sub.ORDER_ID
        FROM
            order_items oi_sub
        WHERE
            oi_sub.PRODUCT_ID = 201)
        AND p.PRODUCT_ID <> 201
        AND a.CITY NOT IN ('Bangalore' , 'New Delhi')
GROUP BY p.PRODUCT_ID , p.PRODUCT_DESC
ORDER BY TOTAL_QUANTITY DESC;

-- 10. WRITE A QUERY TO DISPLAY THE ORDER_ID,CUSTOMER_ID AND CUSTOMER FULLNAME AND TOTAL QUANTITY OF PRODUCTS SHIPPED FOR ORDER IDS 
-- WHICH ARE EVENAND SHIPPED TO ADDRESS WHERE PINCODE IS NOT STARTING WITH "5" 
	-- [NOTE: TABLES TO BE USED - ONLINE_CUSTOMER,ORDER_HEADER, ORDER_ITEMS, ADDRESS]

SELECT 
    oh.ORDER_ID,
    oh.CUSTOMER_ID,
    CASE
        WHEN
            oc.CUSTOMER_GENDER = 'F'
        THEN
            CONCAT('MS. ',
                    UPPER(oc.CUSTOMER_FNAME),
                    ' ',
                    UPPER(oc.CUSTOMER_LNAME))
        ELSE CONCAT('MR. ',
                UPPER(oc.CUSTOMER_FNAME),
                ' ',
                UPPER(oc.CUSTOMER_LNAME))
    END 'CUSTOMER_FULL_NAME',
    COALESCE(SUM(oi.PRODUCT_QUANTITY), 0) TOTAL_QUANTITY
FROM
    order_header oh
        LEFT JOIN
    online_customer oc ON oh.CUSTOMER_ID = oc.CUSTOMER_ID
        LEFT JOIN
    address a ON a.ADDRESS_ID = oc.ADDRESS_ID
        LEFT JOIN
    order_items oi ON oi.ORDER_ID = oh.ORDER_ID
WHERE
    oh.ORDER_ID IN (SELECT 
            ORDER_ID
        FROM
            order_header
        WHERE
            MOD(ORDER_ID, 2) = 0)
        AND a.PINCODE NOT IN (SELECT DISTINCT
            PINCODE
        FROM
            address
        WHERE
            PINCODE LIKE '5%')
GROUP BY 1 , 2 , 3;